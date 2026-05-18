"""
Welcome/Landing Tab - Workflow Selection
Provides clear progressive disclosure for HTR Analysis workflows.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from .theme import Colors, Fonts, Spacing, stylesheet_button_primary


class WorkflowCard(QFrame):
    """Individual workflow selection card with numbered circle."""

    clicked = Signal()

    def __init__(self, number, title, description, accent_color, tooltip_details=""):
        super().__init__()
        self.setup_ui(number, title, description, accent_color, tooltip_details)

    def setup_ui(self, number, title, description, accent_color, tooltip_details):
        """Set up the card UI."""
        self.setFrameStyle(QFrame.NoFrame)
        self.setStyleSheet(f"""
            WorkflowCard {{
                background-color: {Colors.WHITE};
                border-left: 4px solid {accent_color};
                border: 1px solid {Colors.BORDER};
                border-left: 4px solid {accent_color};
                border-radius: 6px;
                padding: 0px;
            }}
            WorkflowCard:hover {{
                background-color: {Colors.LIGHT_BG};
                border-color: {Colors.BORDER_DARK};
                border-left: 4px solid {accent_color};
            }}
            QLabel {{
                background-color: transparent;
                border: none;
            }}
        """)

        if tooltip_details:
            self.setToolTip(tooltip_details)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(12)

        # Numbered circle
        circle = QLabel(str(number))
        circle.setAlignment(Qt.AlignCenter)
        circle.setFixedSize(40, 40)
        circle.setFont(QFont(Fonts.FAMILY, 16, QFont.Bold))
        circle.setStyleSheet(f"""
            background-color: {accent_color};
            color: {Colors.WHITE};
            border-radius: 20px;
            border: none;
        """)
        layout.addWidget(circle)

        # Title + description column
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)

        title_label = QLabel(title)
        title_label.setFont(QFont(Fonts.FAMILY, Fonts.SIZE_HEADER, QFont.Bold))
        text_layout.addWidget(title_label)

        desc_label = QLabel(description)
        desc_label.setFont(QFont(Fonts.FAMILY, Fonts.SIZE_SMALL))
        desc_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        desc_label.setWordWrap(True)
        text_layout.addWidget(desc_label)

        layout.addLayout(text_layout, 1)

        # Go button
        self.button = QPushButton("Go \u2192")
        self.button.setFont(QFont(Fonts.FAMILY, Fonts.SIZE_SMALL, QFont.Bold))
        self.button.setFixedWidth(60)
        self.button.setStyleSheet(f"""
            QPushButton {{
                background-color: {accent_color};
                color: {Colors.WHITE};
                border: none;
                padding: 6px 10px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                opacity: 0.85;
                background-color: {Colors.TEXT_MUTED};
            }}
        """)
        self.button.clicked.connect(self.clicked)
        layout.addWidget(self.button)

        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        """Make entire card clickable."""
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class WelcomeTab(QWidget):
    """Welcome tab with workflow selection cards."""

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
        layout.setContentsMargins(30, 25, 30, 20)
        layout.setSpacing(Spacing.SECTION_SPACING)

        # Header
        title_label = QLabel("HTR Analysis Tool")
        title_label.setFont(QFont(Fonts.FAMILY, Fonts.SIZE_TITLE, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        subtitle_label = QLabel("Choose your workflow to get started")
        subtitle_label.setFont(QFont(Fonts.FAMILY, Fonts.SIZE_SUBHEADER))
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; margin-bottom: 8px;")
        layout.addWidget(subtitle_label)

        # Workflow cards (single column, vertical)
        cards_layout = QVBoxLayout()
        cards_layout.setSpacing(6)

        # Card 1: Tune Parameters
        self.tune_card = WorkflowCard(
            number=1,
            title="Tune Parameters",
            description="Optimize HTR detection parameters with video feedback.",
            accent_color=Colors.ACCENT_1,
            tooltip_details="Requirements: H5 tracking data, video file (optional)\nDifficulty: Advanced\nWhen to use: First time with new data, need to fine-tune detection sensitivity."
        )
        self.tune_card.clicked.connect(self.tune_parameters_requested)
        cards_layout.addWidget(self.tune_card)

        # Connector
        cards_layout.addWidget(self._connector_label())

        # Card 2: Prepare Data
        self.prepare_card = WorkflowCard(
            number=2,
            title="Prepare Data",
            description="Extract features from H5 files and label ground truth data for training.",
            accent_color=Colors.ACCENT_2,
            tooltip_details="Requirements: H5 files, parameters (from step 1 or defaults)\nDifficulty: Intermediate\nWhen to use: Ready to create training data from your H5 files."
        )
        self.prepare_card.clicked.connect(self.prepare_data_requested)
        cards_layout.addWidget(self.prepare_card)

        # Connector
        cards_layout.addWidget(self._connector_label())

        # Card 3: Train Model
        self.train_card = WorkflowCard(
            number=3,
            title="Train Model",
            description="Train XGBoost classifier, evaluate performance, and iteratively refine.",
            accent_color=Colors.ACCENT_3,
            tooltip_details="Requirements: Labeled feature CSVs (from step 2)\nDifficulty: Intermediate\nWhen to use: Have labeled data, need to train or improve model accuracy."
        )
        self.train_card.clicked.connect(self.train_model_requested)
        cards_layout.addWidget(self.train_card)

        # Connector
        cards_layout.addWidget(self._connector_label())

        # Card 4: Deploy
        self.deploy_card = WorkflowCard(
            number=4,
            title="Identify Headtwitches",
            description="Process data using trained models. Supports fresh batches and incremental updates.",
            accent_color=Colors.ACCENT_4,
            tooltip_details="Requirements: H5 files, trained model (.joblib)\nDifficulty: Beginner\nWhen to use: Production analysis - process new data or add to existing project."
        )
        self.deploy_card.clicked.connect(self.deploy_requested)
        cards_layout.addWidget(self.deploy_card)

        layout.addLayout(cards_layout)

        layout.addStretch()

        # Footer
        footer_label = QLabel(
            "Complete workflow: 1 \u2192 2 \u2192 3 \u2192 4  |  "
            "Quick start: Jump to step 4 with a trained model"
        )
        footer_label.setFont(QFont(Fonts.FAMILY, Fonts.SIZE_SMALL))
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setStyleSheet(
            f"color: {Colors.TEXT_MUTED}; "
            f"background-color: {Colors.LIGHT_BG}; "
            f"padding: 8px; border-radius: 4px;"
        )
        layout.addWidget(footer_label)

        # Ensure cards expand horizontally
        for card in [self.tune_card, self.prepare_card, self.train_card, self.deploy_card]:
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def _connector_label(self):
        """Create a subtle vertical connector arrow between cards."""
        label = QLabel("\u2193")
        label.setAlignment(Qt.AlignCenter)
        label.setFont(QFont(Fonts.FAMILY, Fonts.SIZE_SUBHEADER))
        label.setStyleSheet(f"color: {Colors.BORDER_DARK};")
        label.setFixedHeight(18)
        return label
