"""
Welcome/Landing Tab - Workflow Selection
Provides clear progressive disclosure for HTR Analysis workflows.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSizePolicy, QApplication
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPalette


class WorkflowCard(QFrame):
    """Individual workflow selection card."""
    
    clicked = Signal()
    
    def __init__(self, title, description, requirements, difficulty, use_case, icon_text="📊"):
        super().__init__()
        self.setup_ui(title, description, requirements, difficulty, use_case, icon_text)
        
    def setup_ui(self, title, description, requirements, difficulty, use_case, icon_text):
        """Set up the card UI."""
        self.setFrameStyle(QFrame.Box)
        self.setLineWidth(2)
        self.setStyleSheet("""
            WorkflowCard {
                background-color: #f8f9fa;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                padding: 8px;
            }
            WorkflowCard:hover {
                background-color: #e9ecef;
            }
            QLabel {
                background-color: transparent;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(8)
        
        # Icon and title row
        header_layout = QHBoxLayout()
        
        icon_label = QLabel(icon_text)
        icon_label.setFont(QFont("Arial", 24))
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setMaximumWidth(50)
        icon_label.setMinimumWidth(50)
        header_layout.addWidget(icon_label)
        
        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setWordWrap(True)
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Difficulty indicator
        diff_label = QLabel(f"Difficulty: {difficulty}")
        diff_label.setFont(QFont("Arial", 9))
        diff_colors = {
            "Beginner": "#28a745",
            "Intermediate": "#ffc107", 
            "Advanced": "#dc3545"
        }
        color = diff_colors.get(difficulty, "#6c757d")
        diff_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        header_layout.addWidget(diff_label)
        
        layout.addLayout(header_layout)
        
        # Description
        desc_label = QLabel(description)
        desc_label.setFont(QFont("Arial", 10))
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignTop)
        layout.addWidget(desc_label)
        
        # Requirements
        req_label = QLabel(f"<b>You'll need:</b> {requirements}")
        req_label.setFont(QFont("Arial", 9))
        req_label.setWordWrap(True)
        req_label.setStyleSheet("color: #6c757d;")
        layout.addWidget(req_label)
        
        # Use case
        use_case_label = QLabel(f"<b>When to use:</b> {use_case}")
        use_case_label.setFont(QFont("Arial", 9))
        use_case_label.setWordWrap(True)
        use_case_label.setStyleSheet("color: #495057; background-color: #e3f2fd; padding: 6px; border-radius: 3px; margin-top: 4px;")
        layout.addWidget(use_case_label)
        
        # Spacer to push button to bottom
        layout.addStretch()
        
        # Action button
        self.button = QPushButton("Get Started")
        self.button.setFont(QFont("Arial", 10, QFont.Bold))
        self.button.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004085;
            }
        """)
        self.button.clicked.connect(self.clicked)
        layout.addWidget(self.button)
        
        # Make card clickable
        self.setCursor(Qt.PointingHandCursor)
        
    def mousePressEvent(self, event):
        """Make entire card clickable."""
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class WelcomeTab(QWidget):
    """Welcome tab with workflow selection cards."""

    # Signals for workflow selection
    tune_parameters_requested = Signal()
    prepare_data_requested = Signal()
    train_model_requested = Signal()
    deploy_requested = Signal()
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """Initialize the welcome tab UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Header
        header_layout = QVBoxLayout()
        
        title_label = QLabel("HTR Analysis Tool")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title_label)
        
        subtitle_label = QLabel("Choose your workflow to get started")
        subtitle_label.setFont(QFont("Arial", 12))
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("color: #6c757d; margin-bottom: 10px;")
        header_layout.addWidget(subtitle_label)
        
        layout.addLayout(header_layout)
        
        # Workflow cards (in workflow order)
        cards_layout = QVBoxLayout()
        cards_layout.setSpacing(10)

        # Row 1: Tune Parameters
        row1_layout = QHBoxLayout()
        self.tune_card = WorkflowCard(
            title="1. Tune Parameters",
            description="Optimize HTR detection parameters with video feedback. Good starting point for new datasets.",
            requirements="H5 tracking data, video file (optional)",
            difficulty="Advanced",
            use_case="First time with new data type, need to fine-tune detection sensitivity.",
            icon_text="🔧"
        )
        self.tune_card.clicked.connect(self.tune_parameters_requested)
        row1_layout.addWidget(self.tune_card)
        cards_layout.addLayout(row1_layout)

        # Row 2: Prepare Data & Train Model
        row2_layout = QHBoxLayout()
        row2_layout.setSpacing(15)

        self.prepare_card = WorkflowCard(
            title="2. Prepare Data",
            description="Extract features from H5 files and label ground truth data for training.",
            requirements="H5 files, parameters (from step 1 or defaults)",
            difficulty="Intermediate",
            use_case="Ready to create training data from your H5 files.",
            icon_text="📊"
        )
        self.prepare_card.clicked.connect(self.prepare_data_requested)
        row2_layout.addWidget(self.prepare_card)

        self.train_card = WorkflowCard(
            title="3. Train Model",
            description="Train XGBoost classifier, evaluate performance, and iteratively refine.",
            requirements="Labeled feature CSVs (from step 2)",
            difficulty="Intermediate",
            use_case="Have labeled data, need to train or improve model accuracy.",
            icon_text="🧠"
        )
        self.train_card.clicked.connect(self.train_model_requested)
        row2_layout.addWidget(self.train_card)

        cards_layout.addLayout(row2_layout)

        # Row 3: Deploy
        row3_layout = QHBoxLayout()
        self.deploy_card = WorkflowCard(
            title="4. Deploy",
            description="Process data using trained models. Supports both fresh batches and incremental updates.",
            requirements="H5 files, trained model (.joblib)",
            difficulty="Beginner",
            use_case="Production analysis - process new data or add to existing project.",
            icon_text="🚀"
        )
        self.deploy_card.clicked.connect(self.deploy_requested)
        row3_layout.addWidget(self.deploy_card)
        cards_layout.addLayout(row3_layout)

        layout.addLayout(cards_layout)

        # Footer with helpful info
        footer_layout = QVBoxLayout()

        help_label = QLabel(
            "💡 <b>Complete workflow:</b> Tune Parameters → Prepare Data → Train Model → Deploy<br>"
            "⚡ <b>Quick start:</b> If you have a trained model, jump directly to Deploy"
        )
        help_label.setFont(QFont("Arial", 9))
        help_label.setAlignment(Qt.AlignCenter)
        help_label.setStyleSheet("color: #495057; background-color: #f8f9fa; padding: 8px; border-radius: 4px;")
        help_label.setWordWrap(True)
        footer_layout.addWidget(help_label)

        layout.addLayout(footer_layout)

        # Ensure cards have equal size
        for card in [self.tune_card, self.prepare_card, self.train_card, self.deploy_card]:
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)