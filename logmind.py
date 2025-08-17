import sys
import os
import json
import re
from pathlib import Path
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import openai
from openai import OpenAI, OpenAIError
import httpx

class AIAnalysisWorker(QThread):
    """AI分析工作线程，用于异步执行AI模型调用"""
    
    # 定义信号
    analysis_finished = pyqtSignal(str)  # 分析完成信号，携带结果
    analysis_error = pyqtSignal(str)      # 分析错误信号，携带错误信息
    status_update = pyqtSignal(str)       # 状态更新信号
    
    def __init__(self, config, prompt):
        """初始化工作线程
        
        Args:
            config: AI配置
            prompt: 提示词
        """
        super().__init__()
        self.config = config
        self.prompt = prompt
        self._is_running = True
        
    def run(self):
        """线程执行函数"""
        try:
            if not self._is_running:
                return
                
            # 执行AI分析
            result = self._call_ai_model_sync(self.prompt)
            if self._is_running:
                self.analysis_finished.emit(result)
                    
        except Exception as e:
            if self._is_running:
                self.analysis_error.emit(f"AI调用失败：{str(e)}")
    
    def _call_ai_model_sync(self, prompt):
        """同步调用AI模型"""
        try:
            if not self._is_running:
                return "分析已终止"
                
            ai_config = self.config["ai_config"]
            
            if ai_config["model_type"] == "local":
                config = ai_config["local"]
            else:
                config = ai_config["remote"]
            
            # 获取代理配置
            proxy_config = self.config.get("proxy_config", {})
            http_client = None
            
            # 如果代理启用，创建带代理的 httpx 客户端
            if proxy_config.get("enabled", False):
                proxy_host = proxy_config.get("host", "")
                proxy_port = proxy_config.get("port", "")
                proxy_username = proxy_config.get("username", "")
                proxy_password = proxy_config.get("password", "")
                
                if proxy_host and proxy_port:
                    # 构建代理 URL
                    if proxy_username and proxy_password:
                        proxy_url = f"http://{proxy_username}:{proxy_password}@{proxy_host}:{proxy_port}"
                    else:
                        proxy_url = f"http://{proxy_host}:{proxy_port}"
                    
                    # 创建带代理的 httpx 客户端
                    http_client = httpx.Client(proxy=proxy_url)
            
            # 创建 OpenAI 客户端，如果有代理配置则传入 http_client
            if http_client:
                client = OpenAI(
                    base_url=config["base_url"],
                    api_key=config["api_key"],
                    http_client=http_client
                )
            else:
                client = OpenAI(
                    base_url=config["base_url"],
                    api_key=config["api_key"]
                )
            
            # 发送状态更新
            self.status_update.emit("正在调用AI模型...")
            
            # 获取分析参数
            analysis_params = self.config.get("ai_config", {}).get("analysis_params", {})
            temperature = analysis_params.get("temperature", 0.1)
            max_tokens = analysis_params.get("max_tokens", 2000)
            
            response = client.chat.completions.create(
                model=config["model_name"],
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if not self._is_running:
                return "分析已终止"
                
            return response.choices[0].message.content
            
        except Exception as e:
            return f"AI调用失败：{str(e)}\n\n请检查AI模型配置是否正确。"
    
    def stop(self):
        """停止分析"""
        self._is_running = False

class LogMindGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = self.load_config()
        self.context = {
            "log": "",
            "problem_description": "",
            "code_files": {}
        }
        self.analysis_running = False  # 跟踪分析状态
        self.ai_worker = None  # AI分析工作线程
        self.init_ui()
        
        # 连接工作线程信号槽
        self.setup_worker_connections()
        
    def load_config(self):
        """加载配置文件"""
        default_config = {
            "ai_config": {
                "model_type": "local",
                "local": {
                    "base_url": "http://localhost:11434/v1",
                    "api_key": "sk-no-key-required",
                    "model_name": "qwen:14b"
                },
                "remote": {
                    "base_url": "https://api.openai.com/v1",
                    "api_key": "",
                    "model_name": "gpt-4-turbo"
                },
                "analysis_params": {
                    "temperature": 0.1,
                    "max_tokens": 2000
                }
            },
            "ui_config": {
                "problem_description": {
                    "enabled": True,
                    "placeholder_text": "请描述您遇到的问题，包括问题发生的场景、频率、影响范围等信息...",
                    "min_height": 150,
                    "max_length": 2000
                },
                "log_input": {
                    "min_height": 600,
                    "max_length": 10000
                }
            },
            "analysis_config": {
                "input_weights": {
                    "problem_description": 0.4,
                    "log": 0.4,
                    "code": 0.2
                },
                "analysis_guidance": {
                    "with_description_and_log": "请结合问题描述和日志信息进行综合分析，重点关注问题描述中提到的场景和日志中的异常之间的关联。",
                    "with_description_only": "由于没有提供日志信息，请主要基于问题描述进行分析，并建议用户提供相关的错误日志以获得更准确的分析。",
                    "with_log_only": "由于没有提供问题描述，请主要基于日志信息进行分析，并建议用户提供更多关于问题发生场景的描述。"
                }
            },
            "proxy_config": {
                "enabled": False,
                "host": "",
                "port": "",
                "username": "",
                "password": ""
            }
        }
        
        config_file = "logmind_config.json"
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    return {**default_config, **json.load(f)}
            except:
                return default_config
        else:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            return default_config
    
    def save_config(self):
        """保存配置文件"""
        with open("logmind_config.json", 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def init_ui(self):
        """初始化UI界面"""
        self.setWindowTitle("LogMind - 本地日志分析助手")
        self.setGeometry(100, 100, 1400, 900)  # 扩大窗口尺寸
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 标题
        title_label = QLabel("LogMind - 本地日志分析助手")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 2px;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 创建水平分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧区域 - 日志输入区域
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        log_group = QGroupBox("错误日志输入")
        log_layout = QVBoxLayout()
        
        # 添加问题描述框
        problem_desc_label = QLabel("问题描述：")
        problem_desc_label.setStyleSheet("font-weight: bold;")
        log_layout.addWidget(problem_desc_label)
        
        self.problem_desc_text = QTextEdit()
        # 获取问题描述框配置
        problem_desc_config = self.config.get("ui_config", {}).get("problem_description", {})
        placeholder_text = problem_desc_config.get("placeholder_text", "请描述您遇到的问题，包括问题发生的场景、频率、影响范围等信息...")
        min_height = problem_desc_config.get("min_height", 150)
        
        self.problem_desc_text.setPlaceholderText(placeholder_text)
        self.problem_desc_text.setMinimumHeight(min_height)
        log_layout.addWidget(self.problem_desc_text)
        
        # 添加一个分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        log_layout.addWidget(line)
        
        # 添加日志输入标签
        log_label = QLabel("错误日志：")
        log_label.setStyleSheet("font-weight: bold;")
        log_layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        # 获取日志输入框配置
        log_input_config = self.config.get("ui_config", {}).get("log_input", {})
        min_height = log_input_config.get("min_height", 600)
        
        self.log_text.setPlaceholderText("请粘贴您的错误日志（支持多行）\n例如：\n2025-04-05 10:32:15 ERROR [UserService] - Update failed\njava.lang.NullPointerException: Cannot invoke \"String.trim()\" because \"email\" is null\n    at com.example.service.UserService.updateProfile(UserService.java:123)")
        self.log_text.setMinimumHeight(min_height)
        log_layout.addWidget(self.log_text)
        
        # 添加一个分隔线
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setFrameShadow(QFrame.Sunken)
        log_layout.addWidget(line2)
        
        # 添加代码文件夹选择标签和帮助按钮
        folder_header_layout = QHBoxLayout()
        
        code_folder_label = QLabel("相关代码文件夹：")
        code_folder_label.setStyleSheet("font-weight: bold;")
        folder_header_layout.addWidget(code_folder_label)
        
        # 添加帮助按钮
        help_btn = QPushButton("?")
        help_btn.setMaximumWidth(20)
        help_btn.setToolTip("点击查看使用说明")
        help_btn.clicked.connect(self.show_folder_help)
        folder_header_layout.addWidget(help_btn)
        
        folder_header_layout.addStretch()
        log_layout.addLayout(folder_header_layout)
        
        # 添加文件夹使用提示文本
        folder_hint_label = QLabel("提示：添加包含相关代码的文件夹，系统将自动搜索与日志中错误相关的代码文件进行分析。")
        folder_hint_label.setStyleSheet("color: #666; font-size: 11px; margin-bottom: 5px;")
        log_layout.addWidget(folder_hint_label)
        
        # 代码文件夹选择布局
        folder_layout = QHBoxLayout()
        
        # 文件夹列表
        self.folder_list = QListWidget()
        self.folder_list.setMaximumHeight(100)
        folder_layout.addWidget(self.folder_list)
        
        # 文件夹操作按钮
        folder_btn_layout = QVBoxLayout()
        
        self.add_folder_btn = QPushButton("+")
        self.add_folder_btn.setMaximumWidth(30)
        self.add_folder_btn.setToolTip("添加代码文件夹：点击选择包含相关代码的文件夹")
        self.add_folder_btn.clicked.connect(self.add_code_folder)
        folder_btn_layout.addWidget(self.add_folder_btn)
        
        self.remove_folder_btn = QPushButton("-")
        self.remove_folder_btn.setMaximumWidth(30)
        self.remove_folder_btn.setToolTip("移除选中文件夹：从列表中移除选中的代码文件夹")
        self.remove_folder_btn.clicked.connect(self.remove_code_folder)
        folder_btn_layout.addWidget(self.remove_folder_btn)
        
        folder_btn_layout.addStretch()
        folder_layout.addLayout(folder_btn_layout)
        
        log_layout.addLayout(folder_layout)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        self.analyze_btn = QPushButton("🔍 分析日志")
        self.analyze_btn.clicked.connect(self.analyze_log)
        self.clear_btn = QPushButton("🗑️ 清空")
        self.clear_btn.clicked.connect(self.clear_log)
        self.settings_btn = QPushButton("⚙️ 设置")
        self.settings_btn.clicked.connect(self.show_settings)
        
        button_layout.addWidget(self.analyze_btn)
        button_layout.addWidget(self.clear_btn)
        button_layout.addWidget(self.settings_btn)
        button_layout.addStretch()
        
        log_layout.addLayout(button_layout)
        log_group.setLayout(log_layout)
        left_layout.addWidget(log_group)
        
        # 右侧区域 - 分析结果区域
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        result_group = QGroupBox("分析结果")
        result_layout = QVBoxLayout()
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(800)  # 增加高度
        self.result_text.setPlaceholderText("分析结果将显示在这里...")
        result_layout.addWidget(self.result_text)
        
        # 结果操作按钮
        result_button_layout = QHBoxLayout()
        self.copy_btn = QPushButton("📋 复制报告")
        self.copy_btn.clicked.connect(self.copy_report)
        self.export_btn = QPushButton("💾 导出报告")
        self.export_btn.clicked.connect(self.export_report)
        self.stop_analysis_btn = QPushButton("⏹️ 终止分析")
        self.stop_analysis_btn.clicked.connect(self.stop_analysis)
        self.stop_analysis_btn.setEnabled(False)  # 初始状态为禁用
        
        result_button_layout.addWidget(self.copy_btn)
        result_button_layout.addWidget(self.export_btn)
        result_button_layout.addWidget(self.stop_analysis_btn)
        result_button_layout.addStretch()
        
        result_layout.addLayout(result_button_layout)
        result_group.setLayout(result_layout)
        right_layout.addWidget(result_group)
        
        # 将左右部件添加到分割器
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        
        # 设置分割器比例（左侧40%，右侧60%）
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        splitter.setSizes([560, 840])  # 初始尺寸
        
        main_layout.addWidget(splitter)
        
        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")
    
    def setup_worker_connections(self):
        """设置工作线程的信号槽连接"""
        # 这些连接将在创建工作线程时动态建立
        pass
    
    def clear_log(self):
        """清空日志"""
        self.log_text.clear()
        self.problem_desc_text.clear()
        self.result_text.clear()
        self.folder_list.clear()
        self.context = {
            "log": "",
            "problem_description": "",
            "code_files": {}
        }
        self.status_bar.showMessage("已清空")
    
    def show_settings(self):
        """显示设置对话框"""
        dialog = SettingsDialog(self.config)
        if dialog.exec_() == QDialog.Accepted:
            self.config = dialog.get_config()
            self.save_config()
            self.status_bar.showMessage("设置已保存")
    
    def add_code_folder(self):
        """添加代码文件夹"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "选择代码文件夹",
            ".",
            QFileDialog.ShowDirsOnly
        )
        
        if folder_path:
            # 检查是否已经添加过
            for i in range(self.folder_list.count()):
                item_text = self.folder_list.item(i).text()
                # 检查是否只是路径部分相同（去除文件数量信息）
                if item_text.split(" (")[0] == folder_path:
                    QMessageBox.warning(self, "警告", "该文件夹已经添加过了！")
                    return
            
            # 验证文件夹是否存在
            if not os.path.exists(folder_path):
                QMessageBox.warning(self, "错误", f"选择的文件夹不存在！\n路径：{folder_path}")
                return
            
            # 验证是否为文件夹
            if not os.path.isdir(folder_path):
                QMessageBox.warning(self, "错误", f"选择的路径不是文件夹！\n路径：{folder_path}")
                return
            
            # 验证文件夹是否可访问
            if not os.access(folder_path, os.R_OK):
                QMessageBox.warning(self, "错误", f"无法访问该文件夹，请检查权限！\n路径：{folder_path}")
                return
            
            # 统计文件夹中的代码文件数量
            code_file_count = self.count_code_files(folder_path)
            
            # 验证通过，添加到列表，包含文件数量信息
            display_text = f"{folder_path} (包含 {code_file_count} 个代码文件)"
            self.folder_list.addItem(display_text)
            self.status_bar.showMessage(f"已添加代码文件夹：{folder_path}")
    
    def count_code_files(self, folder_path):
        """统计文件夹中的代码文件数量"""
        count = 0
        try:
            folder_path_obj = Path(folder_path)
            # 支持的编程语言文件扩展名
            code_extensions = {".java", ".py", ".js", ".ts", ".cpp", ".c", ".cs", ".go", ".php", ".rb", ".swift", ".kt", ".rs"}
            # 递归搜索所有代码文件
            for file_path in folder_path_obj.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in code_extensions:
                    count += 1
        except Exception as e:
            print(f"统计代码文件时出错：{e}")
        return count
    
    def show_folder_help(self):
        """显示文件夹选择功能的使用说明"""
        help_text = """
        <h3>代码文件夹选择功能使用说明</h3>
        <p><b>功能概述：</b></p>
        <p>此功能允许您添加包含相关代码的文件夹，系统将自动搜索与日志中错误相关的代码文件进行分析。</p>
        
        <p><b>使用步骤：</b></p>
        <ol>
            <li>点击 <b>+</b> 按钮添加代码文件夹</li>
            <li>在弹出的对话框中选择包含相关代码的文件夹</li>
            <li>系统会自动验证文件夹的有效性并添加到列表中</li>
            <li>如需移除文件夹，先选中该文件夹，然后点击 <b>-</b> 按钮</li>
        </ol>
        
        <p><b>注意事项：</b></p>
        <ul>
            <li>请确保添加的文件夹包含与日志错误相关的代码文件</li>
            <li>系统支持递归搜索子文件夹中的代码文件</li>
            <li>文件夹必须具有读取权限</li>
            <li>添加多个文件夹可以提高找到相关代码的概率</li>
        </ul>
        
        <p><b>文件类型支持：</b></p>
        <p>目前支持多种编程语言文件的搜索和分析，包括 Java (.java)、Python (.py)、JavaScript (.js)、TypeScript (.ts)、C++ (.cpp)、C (.c)、C# (.cs)、Go (.go)、PHP (.php)、Ruby (.rb)、Swift (.swift)、Kotlin (.kt)、Rust (.rs) 等。</p>
        """
        
        msg_box = QMessageBox()
        msg_box.setWindowTitle("使用说明")
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setText(help_text)
        msg_box.exec_()
    
    def remove_code_folder(self):
        """移除选中的代码文件夹"""
        selected_items = self.folder_list.selectedItems()
        if selected_items:
            for item in selected_items:
                self.folder_list.takeItem(self.folder_list.row(item))
        else:
            QMessageBox.warning(self, "警告", "请先选择要移除的文件夹！")
    
    def analyze_log(self):
        """分析日志"""
        log_content = self.log_text.toPlainText().strip()
        problem_description = self.problem_desc_text.toPlainText().strip()
        
        # 检查是否至少提供了问题描述或日志
        if not log_content and not problem_description:
            QMessageBox.warning(self, "警告", "请至少输入问题描述或错误日志！")
            return
        
        self.context["log"] = log_content
        self.context["problem_description"] = problem_description
        self.analysis_running = True  # 设置分析状态为运行中
        self.stop_analysis_btn.setEnabled(True)  # 启用终止按钮
        
        # 根据提供的信息设置状态消息
        if log_content and problem_description:
            self.status_bar.showMessage("正在分析问题和日志...")
        elif log_content:
            self.status_bar.showMessage("正在分析日志...")
        else:
            self.status_bar.showMessage("正在分析问题描述...")
            
        QApplication.processEvents()
        
        try:
            analysis_result = None
            
            # 如果有日志内容，则解析日志
            if log_content:
                self.status_bar.showMessage("正在解析日志...")
                QApplication.processEvents()
                analysis_result = self.parse_log(log_content)
            
            # 检查分析是否被终止
            if not self.analysis_running:
                self.result_text.setPlainText("分析已终止")
                self.status_bar.showMessage("分析已终止")
                return
            
            # 如果有日志分析结果且需要代码，则从用户指定的文件夹中获取代码
            if analysis_result and analysis_result.get("needs_code", False):
                # 检查用户是否指定了代码文件夹
                code_folders = []
                valid_folders = []
                for i in range(self.folder_list.count()):
                    item_text = self.folder_list.item(i).text()
                    # 提取实际文件夹路径（去除文件数量信息）
                    folder_path = item_text.split(" (")[0]
                    code_folders.append(folder_path)
                    
                    # 验证文件夹是否存在
                    if not os.path.exists(folder_path):
                        QMessageBox.warning(self, "错误", f"代码文件夹不存在，已跳过！\n路径：{folder_path}")
                        continue
                    
                    # 验证是否为文件夹
                    if not os.path.isdir(folder_path):
                        QMessageBox.warning(self, "错误", f"路径不是文件夹，已跳过！\n路径：{folder_path}")
                        continue
                    
                    # 验证文件夹是否可访问
                    if not os.access(folder_path, os.R_OK):
                        QMessageBox.warning(self, "错误", f"无法访问代码文件夹，已跳过！\n路径：{folder_path}")
                        continue
                    
                    # 验证通过，添加到有效文件夹列表
                    valid_folders.append(folder_path)
                
                if valid_folders:
                    # 如果有部分文件夹无效，显示提示
                    if len(valid_folders) < len(code_folders):
                        invalid_count = len(code_folders) - len(valid_folders)
                        QMessageBox.information(self, "提示", f"有 {invalid_count} 个文件夹无法访问，已跳过这些文件夹。")
                    
                    # 在所有有效的文件夹中搜索相关文件
                    found_files = []
                    for folder_path in valid_folders:
                        if not self.analysis_running:
                            break
                        self.status_bar.showMessage(f"正在搜索 {folder_path} 中的相关文件...")
                        QApplication.processEvents()
                        files = self.search_code_files(folder_path, analysis_result.get("file", ""))
                        found_files.extend(files)
                    
                    if found_files:
                        # 让用户选择找到的文件
                        if len(found_files) == 1:
                            selected_file_path = found_files[0][0]  # 取元组的第一个元素（路径）
                        else:
                            file_names = [f[1] for f in found_files]
                            choice, ok = QInputDialog.getItem(
                                self,
                                "选择代码文件",
                                "找到多个匹配文件，请选择：",
                                file_names,
                                0,
                                False
                            )
                            if ok:
                                selected_file_path = next(f[0] for f in found_files if f[1] == choice)  # 取元组的第一个元素
                            else:
                                selected_file_path = None
                        
                        if selected_file_path and self.read_code_file(selected_file_path, analysis_result.get("line", 123)):
                            # 检查分析状态
                            if not self.analysis_running:
                                self.result_text.setPlainText("分析已终止")
                                self.status_bar.showMessage("分析已终止")
                                return
                                
                            self.status_bar.showMessage("正在综合分析...")
                            QApplication.processEvents()
                            
                            # 直接进行最终分析，不再有AI询问环节
                            final_result = self.final_analysis()
                            self.result_text.setPlainText(final_result)
                            self.status_bar.showMessage("分析完成")
                        else:
                            QMessageBox.warning(self, "错误", "无法读取指定的代码文件！")
                    else:
                        QMessageBox.warning(self, "警告", f"在指定的文件夹中未找到相关文件：{analysis_result.get('file', '')}")
                        # 尝试仅用日志分析
                        final_result = self.final_analysis()
                        self.result_text.setPlainText(final_result)
                        self.status_bar.showMessage("分析完成")
                else:
                    # 所有文件夹都无效，显示提示
                    QMessageBox.warning(self, "警告", "所有添加的代码文件夹都无法访问，将仅基于日志信息进行分析。")
                    # 尝试仅用日志分析
                    final_result = self.final_analysis()
                    self.result_text.setPlainText(final_result)
                    self.status_bar.showMessage("分析完成")
            else:
                # 不需要代码或没有日志，直接分析
                # 检查分析状态
                if not self.analysis_running:
                    self.result_text.setPlainText("分析已终止")
                    self.status_bar.showMessage("分析已终止")
                    return
                    
                self.status_bar.showMessage("正在分析问题...")
                QApplication.processEvents()
                final_result = self.final_analysis()
                self.result_text.setPlainText(final_result)
                self.status_bar.showMessage("分析完成")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"分析过程中发生错误：{str(e)}")
            self.status_bar.showMessage("分析失败")
        finally:
            # 注意：工作线程的清理现在由 _on_analysis_finished 和 _on_analysis_error 方法处理
            pass
    
    def stop_analysis(self):
        """终止分析"""
        self.analysis_running = False
        self.stop_analysis_btn.setEnabled(False)
        
        # 停止AI工作线程
        if self.ai_worker and self.ai_worker.isRunning():
            self.ai_worker.stop()
            self.ai_worker.wait(1000)  # 等待最多1秒让线程停止
            
        self.status_bar.showMessage("分析已终止")
        self.result_text.setPlainText("分析已终止")
    
    def parse_log(self, log_content):
        """解析日志，提取关键信息"""
        # 检查分析状态
        if not self.analysis_running:
            return None
        
        # 简化的日志解析逻辑
        lines = log_content.split('\n')
        analysis = {
            "exception": "",
            "file": "",
            "line": 0,
            "method": "",
            "needs_code": False
        }
        
        # 提取异常类型
        for line in lines:
            if not self.analysis_running:
                return None
            if "Exception" in line and ":" in line:
                analysis["exception"] = line.split(":")[0].strip()
                break
        
        # 提取堆栈信息
        stack_pattern = r'at ([\w\.$]+)\((.*?):(\d+)\)'
        for line in lines:
            if not self.analysis_running:
                return None
            match = re.search(stack_pattern, line)
            if match:
                full_class = match.group(1)
                file_name = match.group(2)
                line_number = int(match.group(3))
                
                # 取第一个匹配的堆栈作为主要位置
                if not analysis["file"]:
                    analysis["file"] = file_name
                    analysis["line"] = line_number
                    analysis["method"] = full_class.split('.')[-1]
                    analysis["needs_code"] = True
                    break
        
        return analysis
    
    
    def search_code_files(self, folder_path, target_file):
        """在文件夹中搜索相关代码文件"""
        found_files = []
        target_filename = Path(target_file).name
        
        # 支持的编程语言文件扩展名
        code_extensions = {".java", ".py", ".js", ".ts", ".cpp", ".c", ".cs", ".go", ".php", ".rb", ".swift", ".kt", ".rs"}
        
        try:
            folder_path_obj = Path(folder_path)
            # 递归搜索所有代码文件
            for code_file in folder_path_obj.rglob("*"):
                if not self.analysis_running:
                    break
                # 检查是否为文件且扩展名在支持列表中
                if code_file.is_file() and code_file.suffix.lower() in code_extensions:
                    if code_file.name == target_filename:
                        found_files.append((str(code_file), f"{code_file.parent.name}/{code_file.name}"))
                    elif target_filename in str(code_file):
                        found_files.append((str(code_file), str(code_file.relative_to(folder_path_obj))))
        except Exception as e:
            print(f"搜索文件时出错：{e}")
        
        return found_files
    def read_code_file(self, file_path, target_line):
        """读取代码文件 - 确保传入的是字符串路径"""
        try:
            # 确保 file_path 是字符串
            if isinstance(file_path, tuple):
                file_path = file_path[0]  # 如果是元组，取第一个元素（路径）
            
            # 转换为 Path 对象
            file_path_obj = Path(file_path)
            
            # 如果是相对路径，转换为绝对路径
            if not file_path_obj.is_absolute():
                file_path_obj = file_path_obj.resolve()
            
            print(f"尝试读取文件: {file_path_obj}")
            print(f"文件是否存在: {file_path_obj.exists()}")
            
            # 检查文件是否存在
            if not file_path_obj.exists():
                QMessageBox.warning(self, "错误", f"指定的文件不存在！\n路径：{file_path_obj}")
                return False
            
            # 检查是否为文件
            if not file_path_obj.is_file():
                QMessageBox.warning(self, "错误", f"指定路径不是文件！\n路径：{file_path_obj}")
                return False
            
            # 读取文件内容
            with open(file_path_obj, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 提取目标行前后10行
            start = max(0, target_line - 11)
            end = min(len(lines), target_line + 9)
            code_context = ''.join(lines[start:end])
            
            self.context["code_files"][file_path_obj.name] = {
                "path": str(file_path_obj),
                "content": code_context,
                "target_line": target_line
            }
            
            print(f"成功读取文件: {file_path_obj.name}")
            return True
        except Exception as e:
            QMessageBox.critical(self, "错误", f"读取文件时发生错误：{str(e)}")
            return False
    
    def _on_status_update(self, status):
        """状态更新的槽函数"""
        self.status_bar.showMessage(status)
        QApplication.processEvents()
    
    def final_analysis(self):
        """最终分析"""
        try:
            # 检查分析状态
            if not self.analysis_running:
                return "分析已终止"
            
            # 准备AI分析的提示词
            prompt = self.build_analysis_prompt()
            
            # 调用AI模型（异步）
            self.status_bar.showMessage("正在准备调用AI模型...")
            QApplication.processEvents()
            
            # 调用AI模型，实际结果将通过信号槽机制传递
            self.call_ai_model(prompt)
            
            # 返回一个占位符，实际结果将通过信号槽机制传递
            return "AI分析已启动，请稍候..."
        except Exception as e:
            return f"AI分析失败：{str(e)}\n\n原始上下文：\n日志：{self.context['log'][:200]}..."
    
    def build_analysis_prompt(self):
        """构建AI分析提示词"""
        # 根据可用信息构建不同的提示词
        has_problem_description = bool(self.context.get('problem_description', '').strip())
        has_log = bool(self.context.get('log', '').strip())
        has_code = bool(self.context.get('code_files', {}))
        
        # 基础提示词
        prompt = """
你是一个资深软件开发工程师，请结合以下信息分析问题根因。

要求：
1. 不要猜测，仅基于提供的信息推理
2. 如果信息不足，请提出具体问题
3. 最终输出格式：
   - 问题现象
   - 根本原因
   - 代码证据
   - 修复建议
   - 预防措施

"""
        
        # 根据可用信息添加不同的上下文
        if has_problem_description:
            prompt += f"问题描述：\n{self.context['problem_description']}\n\n"
        
        if has_log:
            prompt += f"日志信息：\n{self.context['log']}\n\n"
        
        if has_code:
            prompt += "代码信息：\n"
            for filename, code_info in self.context["code_files"].items():
                prompt += f"\n文件 {filename}:\n{code_info['content']}\n"
            prompt += "\n"
        
        # 根据可用信息提供特定的分析指导
        analysis_config = self.config.get("analysis_config", {})
        analysis_guidance = analysis_config.get("analysis_guidance", {})
        
        if has_problem_description and has_log:
            guidance = analysis_guidance.get("with_description_and_log", "请结合问题描述和日志信息进行综合分析，重点关注问题描述中提到的场景和日志中的异常之间的关联。")
            prompt += f"{guidance}\n"
        elif has_problem_description and not has_log:
            guidance = analysis_guidance.get("with_description_only", "由于没有提供日志信息，请主要基于问题描述进行分析，并建议用户提供相关的错误日志以获得更准确的分析。")
            prompt += f"{guidance}\n"
        elif not has_problem_description and has_log:
            guidance = analysis_guidance.get("with_log_only", "由于没有提供问题描述，请主要基于日志信息进行分析，并建议用户提供更多关于问题发生场景的描述。")
            prompt += f"{guidance}\n"
        
        return prompt
    
    def call_ai_model(self, prompt):
        """调用AI模型"""
        # 检查分析状态
        if not self.analysis_running:
            return "分析已终止"
        
        # 创建工作线程执行AI分析
        self.ai_worker = AIAnalysisWorker(self.config, prompt)
        
        # 连接信号槽
        self.ai_worker.analysis_finished.connect(self._on_analysis_finished)
        self.ai_worker.analysis_error.connect(self._on_analysis_error)
        self.ai_worker.status_update.connect(self._on_status_update)
        
        # 启动工作线程
        self.ai_worker.start()
        
        # 不再等待工作线程完成，让其异步运行
        # 返回一个占位符，实际结果将通过信号槽机制传递
        return "AI分析进行中..."
    def _on_analysis_error(self, error_message):
        """分析错误的槽函数"""
        # 更新分析状态
        self.analysis_running = False
        self.stop_analysis_btn.setEnabled(False)
        
        # 显示错误信息
        QMessageBox.critical(self, "分析错误", f"AI分析失败：{error_message}\n\n原始上下文：\n日志：{self.context['log'][:200]}...")
        self.result_text.setPlainText(f"AI分析失败：{error_message}")
        self.status_bar.showMessage("分析失败")
        
        # 清理工作线程
        if self.ai_worker:
            self.ai_worker.deleteLater()
            self.ai_worker = None
    
    def _on_analysis_finished(self, result):
        """分析完成的槽函数"""
        # 更新分析状态
        self.analysis_running = False
        self.stop_analysis_btn.setEnabled(False)
        
        # 显示分析结果
        self.result_text.setPlainText(result)
        self.status_bar.showMessage("分析完成")
        
        # 清理工作线程
        if self.ai_worker:
            self.ai_worker.deleteLater()
            self.ai_worker = None
    
    def copy_report(self):
        """复制报告"""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.result_text.toPlainText())
        self.status_bar.showMessage("报告已复制到剪贴板")
    
    def export_report(self):
        """导出报告"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出报告",
            "logmind_report.txt",
            "文本文件 (*.txt);;Markdown文件 (*.md)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.result_text.toPlainText())
                QMessageBox.information(self, "成功", "报告已成功导出！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败：{str(e)}")
    

class SettingsDialog(QDialog):
    def __init__(self, config):
        super().__init__()
        self.config = config.copy()
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("AI模型设置")
        self.setGeometry(200, 200, 500, 500)
        
        layout = QVBoxLayout()
        
        # 模型类型选择
        type_group = QGroupBox("模型类型")
        type_layout = QHBoxLayout()
        self.local_radio = QRadioButton("本地模型 (Ollama)")
        self.remote_radio = QRadioButton("远程模型 (OpenAI协议)")
        
        model_type = self.config["ai_config"]["model_type"]
        if model_type == "local":
            self.local_radio.setChecked(True)
        else:
            self.remote_radio.setChecked(True)
        
        type_layout.addWidget(self.local_radio)
        type_layout.addWidget(self.remote_radio)
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)
        
        # 本地模型设置
        self.local_group = QGroupBox("本地模型设置")
        local_layout = QFormLayout()
        
        self.local_model = QLineEdit(self.config["ai_config"]["local"]["model_name"])
        self.local_url = QLineEdit(self.config["ai_config"]["local"]["base_url"])
        self.local_key = QLineEdit(self.config["ai_config"]["local"]["api_key"])
        self.local_key.setEchoMode(QLineEdit.Password)
        
        local_layout.addRow("模型名称:", self.local_model)
        local_layout.addRow("API地址:", self.local_url)
        local_layout.addRow("API密钥:", self.local_key)
        
        self.local_group.setLayout(local_layout)
        layout.addWidget(self.local_group)
        
        # 远程模型设置
        self.remote_group = QGroupBox("远程模型设置")
        remote_layout = QFormLayout()
        
        self.remote_model = QLineEdit(self.config["ai_config"]["remote"]["model_name"])
        self.remote_url = QLineEdit(self.config["ai_config"]["remote"]["base_url"])
        self.remote_key = QLineEdit(self.config["ai_config"]["remote"]["api_key"])
        self.remote_key.setEchoMode(QLineEdit.Password)
        
        remote_layout.addRow("模型名称:", self.remote_model)
        remote_layout.addRow("API地址:", self.remote_url)
        remote_layout.addRow("API密钥:", self.remote_key)
        
        self.remote_group.setLayout(remote_layout)
        layout.addWidget(self.remote_group)
        
        # AI连接测试按钮
        test_layout = QHBoxLayout()
        self.test_ai_btn = QPushButton("🧪 测试AI连接")
        self.test_ai_btn.clicked.connect(self.test_ai_connection)
        self.test_result_label = QLabel("")
        test_layout.addWidget(self.test_ai_btn)
        test_layout.addWidget(self.test_result_label)
        test_layout.addStretch()
        
        layout.addLayout(test_layout)
        
        # 代理设置
        proxy_group = QGroupBox("代理设置")
        proxy_layout = QVBoxLayout()
        
        # 代理启用选项
        proxy_enable_layout = QHBoxLayout()
        self.proxy_enable_checkbox = QCheckBox("启用代理")
        self.proxy_enable_checkbox.setChecked(self.config["proxy_config"]["enabled"])
        proxy_enable_layout.addWidget(self.proxy_enable_checkbox)
        proxy_enable_layout.addStretch()
        proxy_layout.addLayout(proxy_enable_layout)
        
        # 代理配置表单
        proxy_form_layout = QFormLayout()
        
        self.proxy_host = QLineEdit(self.config["proxy_config"]["host"])
        self.proxy_port = QLineEdit(self.config["proxy_config"]["port"])
        self.proxy_username = QLineEdit(self.config["proxy_config"]["username"])
        self.proxy_password = QLineEdit(self.config["proxy_config"]["password"])
        self.proxy_password.setEchoMode(QLineEdit.Password)
        
        proxy_form_layout.addRow("代理主机:", self.proxy_host)
        proxy_form_layout.addRow("代理端口:", self.proxy_port)
        proxy_form_layout.addRow("用户名:", self.proxy_username)
        proxy_form_layout.addRow("密码:", self.proxy_password)
        
        proxy_layout.addLayout(proxy_form_layout)
        proxy_group.setLayout(proxy_layout)
        layout.addWidget(proxy_group)
        
        # 连接代理启用信号
        self.proxy_enable_checkbox.toggled.connect(self.toggle_proxy_settings)
        
        # 初始化代理设置状态
        self.toggle_proxy_settings()
        
        # 按钮
        button_layout = QHBoxLayout()
        self.ok_btn = QPushButton("保存设置")
        self.cancel_btn = QPushButton("取消")
        
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(self.ok_btn)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
        # 连接信号
        self.local_radio.toggled.connect(self.toggle_model_type)
        self.remote_radio.toggled.connect(self.toggle_model_type)
        
        # 初始化显示状态
        self.toggle_model_type()
    
    def toggle_model_type(self):
        """切换模型类型显示"""
        if self.local_radio.isChecked():
            self.local_group.setVisible(True)
            self.remote_group.setVisible(False)
        else:
            self.local_group.setVisible(False)
            self.remote_group.setVisible(True)
    
    def toggle_proxy_settings(self):
        """切换代理设置显示状态"""
        enabled = self.proxy_enable_checkbox.isChecked()
        self.proxy_host.setEnabled(enabled)
        self.proxy_port.setEnabled(enabled)
        self.proxy_username.setEnabled(enabled)
        self.proxy_password.setEnabled(enabled)
    
    def test_ai_connection(self):
        """测试AI连接"""
        try:
            self.test_result_label.setText("测试中...")
            QApplication.processEvents()
            
            if self.local_radio.isChecked():
                config = {
                    "base_url": self.local_url.text(),
                    "api_key": self.local_key.text(),
                    "model_name": self.local_model.text()
                }
                model_type = "local"
            else:
                config = {
                    "base_url": self.remote_url.text(),
                    "api_key": self.remote_key.text(),
                    "model_name": self.remote_model.text()
                }
                model_type = "remote"
            
            # 获取代理配置
            proxy_config = self.config.get("proxy_config", {})
            http_client = None
            
            # 如果代理启用，创建带代理的 httpx 客户端
            if proxy_config.get("enabled", False):
                proxy_host = proxy_config.get("host", "")
                proxy_port = proxy_config.get("port", "")
                proxy_username = proxy_config.get("username", "")
                proxy_password = proxy_config.get("password", "")
                
                if proxy_host and proxy_port:
                    # 构建代理 URL
                    if proxy_username and proxy_password:
                        proxy_url = f"http://{proxy_username}:{proxy_password}@{proxy_host}:{proxy_port}"
                    else:
                        proxy_url = f"http://{proxy_host}:{proxy_port}"
                    
                    # 创建带代理的 httpx 客户端
                    http_client = httpx.Client(proxy=proxy_url)
            
            # 创建 OpenAI 客户端，如果有代理配置则传入 http_client
            if http_client:
                client = OpenAI(
                    base_url=config["base_url"],
                    api_key=config["api_key"],
                    http_client=http_client
                )
            else:
                client = OpenAI(
                    base_url=config["base_url"],
                    api_key=config["api_key"]
                )
            
            # 发送简单的测试请求
            response = client.chat.completions.create(
                model=config["model_name"],
                messages=[{"role": "user", "content": "Hello, this is a connection test. Please respond with 'Connection successful'."}],
                temperature=0.1,
                max_tokens=50
            )
            
            result = response.choices[0].message.content
            self.test_result_label.setText("✅ 连接成功")
            QMessageBox.information(self, "AI连接测试", f"连接成功！\nAI响应：{result}")
            
        except OpenAIError as e:
            self.test_result_label.setText("❌ 连接失败")
            QMessageBox.critical(self, "AI连接测试失败", f"OpenAI错误：{str(e)}")
        except Exception as e:
            self.test_result_label.setText("❌ 连接失败")
            QMessageBox.critical(self, "AI连接测试失败", f"连接错误：{str(e)}")
    
    def get_config(self):
        """获取配置"""
        self.config["ai_config"]["model_type"] = "local" if self.local_radio.isChecked() else "remote"
        self.config["ai_config"]["local"]["model_name"] = self.local_model.text()
        self.config["ai_config"]["local"]["base_url"] = self.local_url.text()
        self.config["ai_config"]["local"]["api_key"] = self.local_key.text()
        self.config["ai_config"]["remote"]["model_name"] = self.remote_model.text()
        self.config["ai_config"]["remote"]["base_url"] = self.remote_url.text()
        self.config["ai_config"]["remote"]["api_key"] = self.remote_key.text()
        
        # 保存代理配置
        self.config["proxy_config"]["enabled"] = self.proxy_enable_checkbox.isChecked()
        self.config["proxy_config"]["host"] = self.proxy_host.text()
        self.config["proxy_config"]["port"] = self.proxy_port.text()
        self.config["proxy_config"]["username"] = self.proxy_username.text()
        self.config["proxy_config"]["password"] = self.proxy_password.text()
        
        return self.config

class MultiLineInputDialog(QDialog):
    """多行文本输入对话框"""
    def __init__(self, parent=None, title="输入", label="请输入内容:", max_length=2000):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.max_length = max_length
        self.init_ui(label)
        
    def init_ui(self, label_text):
        """初始化UI界面"""
        layout = QVBoxLayout()
        
        # 问题标签
        self.label = QLabel(label_text)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        
        # 多行文本输入框
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("请在此输入您的回答...")
        self.text_edit.setMinimumHeight(150)
        self.text_edit.textChanged.connect(self.on_text_changed)
        layout.addWidget(self.text_edit)
        
        # 字符计数标签
        self.char_count_label = QLabel(f"0 / {self.max_length}")
        self.char_count_label.setAlignment(Qt.AlignRight)
        layout.addWidget(self.char_count_label)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        
        self.ok_btn = QPushButton("确定")
        self.ok_btn.setDefault(True)
        self.ok_btn.clicked.connect(self.accept)
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(self.ok_btn)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
        # 设置对话框大小
        self.setMinimumSize(500, 300)
        self.resize(500, 350)
        
    def on_text_changed(self):
        """文本改变时更新字符计数"""
        text = self.text_edit.toPlainText()
        current_length = len(text)
        self.char_count_label.setText(f"{current_length} / {self.max_length}")
        
        # 如果超过最大长度，截断文本
        if current_length > self.max_length:
            self.text_edit.setPlainText(text[:self.max_length])
            # 将光标移动到末尾
            cursor = self.text_edit.textCursor()
            cursor.setPosition(self.max_length)
            self.text_edit.setTextCursor(cursor)
    
    def get_text(self):
        """获取输入的文本"""
        return self.text_edit.toPlainText().strip()
    
    @staticmethod
    def getText(parent=None, title="输入", label="请输入内容:", max_length=2000):
        """静态方法，显示对话框并返回用户输入的文本"""
        dialog = MultiLineInputDialog(parent, title, label, max_length)
        result = dialog.exec_()
        return (dialog.get_text(), result == QDialog.Accepted)

def main():
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle('Fusion')
    
    # 创建主窗口
    window = LogMindGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()