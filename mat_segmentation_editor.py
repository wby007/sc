import sys
import numpy as np
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QListWidget, QListWidgetItem, QFileDialog,
                             QMessageBox, QGroupBox, QSplitter, QStatusBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.io import loadmat, savemat

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


class MatProcessor:
    """MAT文件处理工具类，沿用验证过的提取逻辑"""

    @staticmethod
    def find_segmentation_mask(mat_data):
        """尝试在MAT文件中查找分割掩码数据（沿用原代码）"""
        common_segmentation_names = ['segmentation', 'seg', 'mask', 'labels', 'gt']

        for var_name, var_data in mat_data.items():
            # 跳过MATLAB系统变量
            if var_name.startswith('__'):
                continue

            # 检查变量是否为数组
            if isinstance(var_data, np.ndarray):
                # 检查是否为结构化数组
                if var_data.dtype.names:
                    # 检查结构化数组的字段
                    for field_name in var_data.dtype.names:
                        if any(name in field_name.lower() for name in common_segmentation_names):
                            try:
                                field_data = var_data[field_name][0, 0]
                                if isinstance(field_data, np.ndarray) and len(field_data.shape) in [2, 3]:
                                    return field_data, f"{var_name}.{field_name}"
                            except:
                                continue
                else:
                    # 普通数组
                    if len(var_data.shape) in [2, 3] and var_data.dtype != object:
                        return var_data, var_name

        return None, None

    @staticmethod
    def custom_segmentation_processor(mat_data):
        """自定义处理器，专门提取分割掩码（沿用原代码逻辑）"""
        try:
            # 尝试标准SBD格式
            if 'GTinst' in mat_data:
                gt_data = mat_data['GTinst'][0, 0]
                if 'Segmentation' in gt_data.dtype.names:
                    seg_mask = gt_data['Segmentation'][0, 0]
                    if isinstance(seg_mask, np.ndarray) and len(seg_mask.shape) in [2, 3]:
                        return seg_mask, "GTinst.Segmentation"

            # 尝试SBD的GTcls格式
            if 'GTcls' in mat_data:
                gt_data = mat_data['GTcls'][0, 0]
                if 'Segmentation' in gt_data.dtype.names:
                    seg_mask = gt_data['Segmentation'][0, 0]
                    if isinstance(seg_mask, np.ndarray) and len(seg_mask.shape) in [2, 3]:
                        return seg_mask, "GTcls.Segmentation"

            # 尝试更简单的结构
            if 'segmentation' in mat_data:
                seg_mask = mat_data['segmentation']
                if isinstance(seg_mask, np.ndarray) and len(seg_mask.shape) in [2, 3]:
                    return seg_mask, "segmentation"

            return None, None
        except Exception as e:
            print(f"自定义处理器出错: {e}")
            return None, None

    @staticmethod
    def extract_classes(seg_mask):
        """从分割掩码中提取类别ID"""
        if seg_mask is None:
            return []

        unique_ids = np.unique(seg_mask)
        # 排除背景(假设0是背景)
        return [id for id in unique_ids if id != 0]


class MatLoader(QThread):
    """后台加载MAT文件的线程"""
    progress_updated = pyqtSignal(str)
    load_finished = pyqtSignal(object, object, list, str)
    load_failed = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.mat_data = None
        self.seg_mask = None
        self.class_ids = []
        self.source_info = ""

    def run(self):
        try:
            self.progress_updated.emit(f"正在加载文件: {os.path.basename(self.file_path)}")

            # 加载MAT文件
            self.mat_data = loadmat(self.file_path)
            self.progress_updated.emit("成功加载MAT文件")

            # 尝试提取分割掩码（使用用户提供的逻辑）
            success = False

            # 1. 尝试自定义处理器（沿用用户代码）
            self.seg_mask, self.source_info = MatProcessor.custom_segmentation_processor(self.mat_data)
            if self.seg_mask is not None:
                self.progress_updated.emit(f"使用自定义处理器提取到分割掩码: {self.source_info}")
                success = True

            # 2. 如果失败，尝试通用方法（沿用用户代码）
            if not success:
                self.progress_updated.emit("尝试通用方法提取分割掩码")
                self.seg_mask, self.source_info = MatProcessor.find_segmentation_mask(self.mat_data)
                if self.seg_mask is not None:
                    self.progress_updated.emit(f"使用通用方法提取到分割掩码: {self.source_info}")
                    success = True

            # 检查是否成功提取
            if not success or self.seg_mask is None:
                self.load_failed.emit("无法从MAT文件中提取有效的分割掩码")
                return

            # 检查掩码有效性
            if not isinstance(self.seg_mask, np.ndarray) or self.seg_mask.ndim not in [2, 3]:
                self.load_failed.emit(f"分割掩码格式无效，形状: {getattr(self.seg_mask, 'shape', '未知')}")
                return

            self.progress_updated.emit(f"提取到分割掩码，形状: {self.seg_mask.shape}, 类型: {self.seg_mask.dtype}")

            # 提取类别ID
            self.class_ids = MatProcessor.extract_classes(self.seg_mask)
            self.progress_updated.emit(f"找到 {len(self.class_ids)} 个类别")

            # 发送加载完成信号
            self.load_finished.emit(self.mat_data, self.seg_mask, self.class_ids, self.source_info)

        except Exception as e:
            self.load_failed.emit(f"加载数据出错: {str(e)}")


class MatSegmentEditor(QMainWindow):
    """MAT文件分割掩码编辑界面"""

    def __init__(self):
        super().__init__()
        self.mat_file = None
        self.mat_data = None
        self.seg_mask = None
        self.original_mask = None
        self.class_ids = []
        self.class_names = []
        self.keep_status = []
        self.source_info = ""

        self.init_ui()

    def init_ui(self):
        """初始化UI界面"""
        self.setWindowTitle("MAT分割掩码类别编辑工具")
        self.setGeometry(100, 100, 1200, 800)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 顶部按钮和信息区
        top_layout = QHBoxLayout()

        # 加载文件按钮
        self.load_btn = QPushButton("加载MAT文件")
        self.load_btn.clicked.connect(self.load_mat_file)
        top_layout.addWidget(self.load_btn)

        # 保存文件按钮
        self.save_btn = QPushButton("保存修改")
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)
        top_layout.addWidget(self.save_btn)

        # 文件信息标签
        self.file_label = QLabel("未加载文件")
        self.file_label.setMinimumWidth(300)
        top_layout.addWidget(self.file_label)

        # 状态标签
        self.status_label = QLabel("就绪")
        top_layout.addWidget(self.status_label, 1)

        main_layout.addLayout(top_layout)

        # 主分割器
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setSizes([300, 900])

        # 左侧面板 - 类别选择
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # 类别选择组
        class_group = QGroupBox("选择要保留的类别")
        class_layout = QVBoxLayout(class_group)

        self.class_list = QListWidget()
        self.class_list.setSelectionMode(QListWidget.NoSelection)
        class_layout.addWidget(self.class_list)

        left_layout.addWidget(class_group)

        # 信息显示
        info_group = QGroupBox("掩码信息")
        info_layout = QVBoxLayout(info_group)
        self.info_text = QLabel("未加载数据")
        self.info_text.setWordWrap(True)
        info_layout.addWidget(self.info_text)
        left_layout.addWidget(info_group)

        # 操作按钮
        btn_layout = QHBoxLayout()

        self.apply_btn = QPushButton("应用选择")
        self.apply_btn.clicked.connect(self.apply_selection)
        self.apply_btn.setEnabled(False)
        btn_layout.addWidget(self.apply_btn)

        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self.reset_selection)
        self.reset_btn.setEnabled(False)
        btn_layout.addWidget(self.reset_btn)

        left_layout.addLayout(btn_layout)

        main_splitter.addWidget(left_panel)

        # 右侧面板 - 图像显示
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 图像显示区域
        img_splitter = QSplitter(Qt.Vertical)
        img_splitter.setSizes([400, 400])

        # 原始图像
        original_group = QGroupBox("原始分割掩码")
        original_layout = QVBoxLayout(original_group)
        self.original_canvas = FigureCanvas(plt.figure(figsize=(6, 4), dpi=100))
        original_layout.addWidget(self.original_canvas)
        img_splitter.addWidget(original_group)

        # 处理后图像
        processed_group = QGroupBox("处理后分割掩码")
        processed_layout = QVBoxLayout(processed_group)
        self.processed_canvas = FigureCanvas(plt.figure(figsize=(6, 4), dpi=100))
        processed_layout.addWidget(self.processed_canvas)
        img_splitter.addWidget(processed_group)

        right_layout.addWidget(img_splitter)
        main_splitter.addWidget(right_panel)

        main_layout.addWidget(main_splitter)

        # 状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("就绪")

    def load_mat_file(self):
        """加载MAT文件对话框"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择MAT文件", "", "MAT文件 (*.mat);;所有文件 (*)"
        )

        if not file_path:
            return

        # 更新界面状态
        self.file_label.setText(os.path.basename(file_path))
        self.status_label.setText("正在加载文件...")
        self.statusBar.showMessage(f"正在加载: {file_path}")

        # 创建并启动加载线程
        self.loader = MatLoader(file_path)
        self.loader.progress_updated.connect(self.update_status)
        self.loader.load_finished.connect(self.on_load_finished)
        self.loader.load_failed.connect(self.on_load_failed)
        self.loader.start()

    def update_status(self, message):
        """更新状态信息"""
        self.status_label.setText(message)
        self.statusBar.showMessage(message)

    def on_load_finished(self, mat_data, seg_mask, class_ids, source_info):
        """加载完成处理"""
        # 保存加载的数据
        self.mat_data = mat_data
        self.mat_file = self.loader.file_path
        self.seg_mask = seg_mask.copy()
        self.original_mask = seg_mask.copy()
        self.class_ids = class_ids
        self.source_info = source_info

        # 生成类别名称
        self.class_names = [f"类别 {id}" for id in self.class_ids]

        # 初始化保留状态（默认保留所有类别）
        self.keep_status = [True] * len(self.class_ids)

        # 更新界面
        self.populate_class_list()
        self.display_images()
        self.update_info_text()

        # 启用按钮
        self.save_btn.setEnabled(True)
        self.apply_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)

        self.update_status(f"成功加载，包含 {len(class_ids)} 个类别")

    def on_load_failed(self, error_message):
        """加载失败处理"""
        QMessageBox.critical(self, "加载失败", error_message)
        self.update_status(f"加载失败: {error_message}")

    def populate_class_list(self):
        """填充类别列表"""
        self.class_list.clear()

        for i, (class_id, class_name) in enumerate(zip(self.class_ids, self.class_names)):
            item = QListWidgetItem(f"ID {class_id}: {class_name}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if self.keep_status[i] else Qt.Unchecked)
            self.class_list.addItem(item)

    def update_info_text(self):
        """更新信息文本"""
        info = [
            f"文件路径: {os.path.basename(self.mat_file)}",
            f"掩码来源: {self.source_info}",
            f"掩码形状: {self.seg_mask.shape}",
            f"数据类型: {self.seg_mask.dtype}",
            f"类别数量: {len(self.class_ids)}"
        ]
        self.info_text.setText("\n".join(info))

    def display_images(self):
        """显示原始和处理后的图像"""
        # 显示原始图像
        self.original_canvas.figure.clear()
        ax_original = self.original_canvas.figure.add_subplot(111)
        ax_original.imshow(self.original_mask, cmap='viridis')
        ax_original.set_title("原始分割掩码")
        ax_original.axis('off')
        self.original_canvas.figure.tight_layout()
        self.original_canvas.draw()

        # 初始显示处理后的图像（与原始相同）
        self.processed_canvas.figure.clear()
        ax_processed = self.processed_canvas.figure.add_subplot(111)
        ax_processed.imshow(self.seg_mask, cmap='viridis')
        ax_processed.set_title("处理后分割掩码")
        ax_processed.axis('off')
        self.processed_canvas.figure.tight_layout()
        self.processed_canvas.draw()

    def apply_selection(self):
        """应用类别选择"""
        # 更新保留状态
        for i in range(self.class_list.count()):
            item = self.class_list.item(i)
            self.keep_status[i] = (item.checkState() == Qt.Checked)

        # 创建处理后的掩码
        processed_mask = np.zeros_like(self.original_mask)

        # 保留选中的类别
        for i, (class_id, keep) in enumerate(zip(self.class_ids, self.keep_status)):
            if keep:
                processed_mask[self.original_mask == class_id] = class_id

        # 更新显示
        self.seg_mask = processed_mask
        self.processed_canvas.figure.clear()
        ax_processed = self.processed_canvas.figure.add_subplot(111)
        ax_processed.imshow(self.seg_mask, cmap='viridis')
        ax_processed.set_title("处理后分割掩码")
        ax_processed.axis('off')
        self.processed_canvas.figure.tight_layout()
        self.processed_canvas.draw()

        self.update_status(f"已应用选择，保留 {sum(self.keep_status)} 个类别")

    def reset_selection(self):
        """重置选择"""
        # 重置保留状态
        self.keep_status = [True] * len(self.class_ids)

        # 重置复选框
        for i in range(self.class_list.count()):
            self.class_list.item(i).setCheckState(Qt.Checked)

        # 重置掩码
        self.seg_mask = self.original_mask.copy()
        self.processed_canvas.figure.clear()
        ax_processed = self.processed_canvas.figure.add_subplot(111)
        ax_processed.imshow(self.seg_mask, cmap='viridis')
        ax_processed.set_title("处理后分割掩码")
        ax_processed.axis('off')
        self.processed_canvas.figure.tight_layout()
        self.processed_canvas.draw()

        self.update_status("已重置所有选择")

    def save_result(self):
        """保存处理结果"""
        if not self.mat_file or self.seg_mask is None:
            QMessageBox.warning(self, "保存失败", "没有可保存的数据")
            return

        # 构建保存路径
        dir_name = os.path.dirname(self.mat_file)
        base_name = os.path.splitext(os.path.basename(self.mat_file))[0]
        save_path = os.path.join(dir_name, f"{base_name}_edited.mat")

        # 询问用户是否确认保存
        reply = QMessageBox.question(
            self, "保存结果",
            f"是否将处理后的分割掩码保存到:\n{save_path}",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )

        if reply != QMessageBox.Yes:
            return

        try:
            # 根据分割掩码来源更新MAT数据
            if self.source_info.startswith("GTinst.") or self.source_info.startswith("GTcls."):
                # 处理GTinst/GTcls结构中的Segmentation字段
                struct_name = self.source_info.split('.')[0]
                field_name = self.source_info.split('.')[1]

                # 确保数据格式正确
                if self.mat_data[struct_name].ndim == 2 and self.mat_data[struct_name].shape == (1, 1):
                    # 标准SBD格式
                    if field_name in self.mat_data[struct_name][0, 0].dtype.names:
                        # 保持原有结构不变，只更新数据
                        original_field = self.mat_data[struct_name][0, 0][field_name]

                        # 检查原始数据的维度结构
                        if original_field.ndim >= 2 and original_field.shape[0] == 1 and original_field.shape[1] == 1:
                            # 嵌套结构: [[data]]
                            self.mat_data[struct_name][0, 0][field_name][0, 0] = self.seg_mask
                        elif original_field.ndim == 2 and original_field.shape[0] == 1:
                            # 结构: [data]
                            self.mat_data[struct_name][0, 0][field_name][0] = self.seg_mask
                        else:
                            # 直接替换
                            self.mat_data[struct_name][0, 0][field_name] = self.seg_mask
                    else:
                        raise ValueError(f"结构 {struct_name} 中找不到字段 {field_name}")
                else:
                    raise ValueError(f"结构 {struct_name} 格式不符合预期")

            else:
                # 直接更新变量
                var_name = self.source_info.split('.')[0]
                self.mat_data[var_name] = self.seg_mask

            # 保存修改后的MAT文件
            savemat(save_path, self.mat_data)

            # 显示保存成功消息
            QMessageBox.information(self, "保存成功", f"已成功保存到:\n{save_path}")
            self.update_status(f"已保存到: {os.path.basename(save_path)}")

        except Exception as e:
            # 显示保存失败消息
            QMessageBox.critical(self, "保存失败", f"保存文件时出错:\n{str(e)}")
            self.update_status(f"保存失败: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MatSegmentEditor()
    window.show()
    sys.exit(app.exec_())