"""
Centralized Theme Module for HTR Analysis Tool v3.

Provides colors, fonts, spacing constants, stylesheet generators,
and icon helpers for consistent UI styling across all tabs.
"""

from PySide6.QtWidgets import QStyle, QApplication
from PySide6.QtGui import QIcon


class Colors:
    """Application color constants."""
    PRIMARY = "#007bff"
    PRIMARY_HOVER = "#0056b3"
    PRIMARY_PRESSED = "#004085"

    SUCCESS = "#28a745"
    SUCCESS_HOVER = "#218838"

    DANGER = "#dc3545"
    DANGER_HOVER = "#c82333"

    WARNING = "#ffc107"
    WARNING_BG = "#fff3cd"
    WARNING_TEXT = "#856404"

    INFO = "#17a2b8"
    INFO_BG = "#d1ecf1"
    INFO_TEXT = "#0c5460"

    LIGHT_BG = "#f8f9fa"
    MEDIUM_BG = "#e9ecef"
    BORDER = "#dee2e6"
    BORDER_DARK = "#ced4da"

    TEXT = "#212529"
    TEXT_SECONDARY = "#6c757d"
    TEXT_MUTED = "#495057"

    SUCCESS_BG = "#d4edda"
    SUCCESS_TEXT = "#155724"

    DISABLED = "#6c757d"

    WHITE = "#ffffff"

    # Card accent colors (for welcome tab workflow cards)
    ACCENT_1 = "#6f42c1"  # Purple - Tune
    ACCENT_2 = "#007bff"  # Blue - Prepare
    ACCENT_3 = "#28a745"  # Green - Train
    ACCENT_4 = "#fd7e14"  # Orange - Deploy


class Fonts:
    """Application font constants."""
    FAMILY = "Segoe UI"
    MONO_FAMILY = "Consolas"

    SIZE_TITLE = 18
    SIZE_HEADER = 14
    SIZE_SUBHEADER = 12
    SIZE_BODY = 10
    SIZE_SMALL = 9
    SIZE_TINY = 8


class Spacing:
    """Application spacing constants."""
    PAGE_MARGIN = 15
    GROUP_MARGIN = 10
    SECTION_SPACING = 15
    ITEM_SPACING = 8
    COMPACT_SPACING = 5


def stylesheet_button_primary():
    """Return stylesheet for primary (blue) buttons."""
    return f"""
        QPushButton {{
            background-color: {Colors.PRIMARY};
            color: {Colors.WHITE};
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {Colors.PRIMARY_HOVER};
        }}
        QPushButton:pressed {{
            background-color: {Colors.PRIMARY_PRESSED};
        }}
        QPushButton:disabled {{
            background-color: {Colors.DISABLED};
        }}
    """


def stylesheet_button_success():
    """Return stylesheet for success (green) buttons."""
    return f"""
        QPushButton {{
            background-color: {Colors.SUCCESS};
            color: {Colors.WHITE};
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {Colors.SUCCESS_HOVER};
        }}
        QPushButton:disabled {{
            background-color: {Colors.DISABLED};
        }}
    """


def stylesheet_button_danger():
    """Return stylesheet for danger (red) buttons."""
    return f"""
        QPushButton {{
            background-color: {Colors.DANGER};
            color: {Colors.WHITE};
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {Colors.DANGER_HOVER};
        }}
        QPushButton:disabled {{
            background-color: {Colors.DISABLED};
        }}
    """


def stylesheet_status_info():
    """Return stylesheet for info status labels."""
    return f"""
        background-color: {Colors.LIGHT_BG};
        padding: 8px;
        border-radius: 4px;
        color: {Colors.TEXT_SECONDARY};
    """


def stylesheet_status_warning():
    """Return stylesheet for warning status labels."""
    return f"""
        background-color: {Colors.WARNING_BG};
        padding: 10px;
        border-radius: 4px;
    """


def stylesheet_status_success():
    """Return stylesheet for success status labels."""
    return f"""
        background-color: {Colors.SUCCESS_BG};
        padding: 8px;
        border-radius: 4px;
        color: {Colors.SUCCESS_TEXT};
    """


def stylesheet_status_dynamic(bg_color):
    """Return stylesheet for dynamically-colored status labels."""
    return f"""
        background-color: {bg_color};
        padding: 10px;
        border-radius: 4px;
    """


def stylesheet_log_area():
    """Return stylesheet for log/progress QTextEdit areas."""
    return f"""
        QTextEdit {{
            background-color: {Colors.LIGHT_BG};
            border: 1px solid {Colors.BORDER};
            color: {Colors.TEXT_MUTED};
        }}
    """


def stylesheet_separator():
    """Return stylesheet for horizontal separator lines."""
    return f"background-color: {Colors.BORDER}; min-height: 2px; max-height: 2px;"


def get_icon(name):
    """
    Get a QIcon from Qt's standard pixmap set.

    Args:
        name: Icon name string. Supported names:
            'open', 'save', 'reset', 'folder', 'play', 'combine',
            'reload', 'computer', 'chart', 'search', 'info',
            'apply', 'cancel', 'help', 'warning', 'critical'

    Returns:
        QIcon from QStyle standard pixmap
    """
    style = QApplication.style()
    icon_map = {
        'open': QStyle.StandardPixmap.SP_DialogOpenButton,
        'save': QStyle.StandardPixmap.SP_DialogSaveButton,
        'reset': QStyle.StandardPixmap.SP_BrowserReload,
        'folder': QStyle.StandardPixmap.SP_DirOpenIcon,
        'play': QStyle.StandardPixmap.SP_MediaPlay,
        'combine': QStyle.StandardPixmap.SP_FileDialogDetailedView,
        'reload': QStyle.StandardPixmap.SP_BrowserReload,
        'computer': QStyle.StandardPixmap.SP_ComputerIcon,
        'chart': QStyle.StandardPixmap.SP_FileDialogDetailedView,
        'search': QStyle.StandardPixmap.SP_FileDialogContentsView,
        'info': QStyle.StandardPixmap.SP_MessageBoxInformation,
        'apply': QStyle.StandardPixmap.SP_DialogApplyButton,
        'cancel': QStyle.StandardPixmap.SP_DialogCancelButton,
        'help': QStyle.StandardPixmap.SP_DialogHelpButton,
        'warning': QStyle.StandardPixmap.SP_MessageBoxWarning,
        'critical': QStyle.StandardPixmap.SP_MessageBoxCritical,
        'file': QStyle.StandardPixmap.SP_FileIcon,
        'new': QStyle.StandardPixmap.SP_FileDialogNewFolder,
    }
    pixmap_enum = icon_map.get(name, QStyle.StandardPixmap.SP_FileIcon)
    return style.standardIcon(pixmap_enum)


def get_app_stylesheet():
    """
    Return a global application stylesheet for consistent widget styling.

    Apply via app.setStyleSheet(get_app_stylesheet()) in main().
    """
    return f"""
        /* ---- QGroupBox ---- */
        QGroupBox {{
            border: 1px solid {Colors.BORDER};
            border-radius: 6px;
            margin-top: 14px;
            padding: 22px 8px 8px 8px;
            font-family: "{Fonts.FAMILY}";
            font-size: {Fonts.SIZE_BODY}pt;
            font-weight: bold;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 2px 8px;
            color: {Colors.TEXT};
        }}

        /* ---- QTabBar ---- */
        QTabBar::tab {{
            font-family: "{Fonts.FAMILY}";
            font-size: {Fonts.SIZE_BODY}pt;
            padding: 8px 18px;
            margin-right: 2px;
            border: none;
            border-bottom: 3px solid transparent;
            background: {Colors.LIGHT_BG};
            color: {Colors.TEXT_SECONDARY};
        }}
        QTabBar::tab:selected {{
            background: {Colors.WHITE};
            color: {Colors.PRIMARY};
            border-bottom: 3px solid {Colors.PRIMARY};
            font-weight: bold;
        }}
        QTabBar::tab:hover:!selected {{
            background: {Colors.MEDIUM_BG};
            color: {Colors.TEXT};
        }}
        QTabWidget::pane {{
            border: 1px solid {Colors.BORDER};
            border-top: none;
        }}

        /* ---- QPushButton (neutral base) ---- */
        QPushButton {{
            font-family: "{Fonts.FAMILY}";
            font-size: {Fonts.SIZE_SMALL}pt;
            padding: 5px 12px;
            border: 1px solid {Colors.BORDER_DARK};
            border-radius: 4px;
            background: {Colors.WHITE};
            color: {Colors.TEXT};
        }}
        QPushButton:hover {{
            background: {Colors.LIGHT_BG};
            border-color: {Colors.TEXT_SECONDARY};
        }}
        QPushButton:pressed {{
            background: {Colors.MEDIUM_BG};
        }}
        QPushButton:disabled {{
            color: {Colors.BORDER_DARK};
            background: {Colors.LIGHT_BG};
            border-color: {Colors.BORDER};
        }}

        /* ---- QLineEdit / QComboBox ---- */
        QLineEdit, QComboBox {{
            font-family: "{Fonts.FAMILY}";
            font-size: {Fonts.SIZE_SMALL}pt;
            padding: 4px 8px;
            border: 1px solid {Colors.BORDER_DARK};
            border-radius: 4px;
            background: {Colors.WHITE};
        }}
        QLineEdit:focus, QComboBox:focus {{
            border-color: {Colors.PRIMARY};
            outline: none;
        }}

        /* ---- QProgressBar ---- */
        QProgressBar {{
            border: 1px solid {Colors.BORDER};
            border-radius: 4px;
            text-align: center;
            font-family: "{Fonts.FAMILY}";
            font-size: {Fonts.SIZE_TINY}pt;
            background: {Colors.LIGHT_BG};
        }}
        QProgressBar::chunk {{
            background-color: {Colors.PRIMARY};
            border-radius: 3px;
        }}

        /* ---- QTableWidget / QHeaderView ---- */
        QTableWidget {{
            gridline-color: {Colors.BORDER};
            font-family: "{Fonts.FAMILY}";
            font-size: {Fonts.SIZE_SMALL}pt;
        }}
        QHeaderView::section {{
            background-color: {Colors.LIGHT_BG};
            border: 1px solid {Colors.BORDER};
            padding: 4px 8px;
            font-family: "{Fonts.FAMILY}";
            font-size: {Fonts.SIZE_SMALL}pt;
            font-weight: bold;
        }}

        /* ---- QLabel default ---- */
        QLabel {{
            font-family: "{Fonts.FAMILY}";
        }}

        /* ---- QCheckBox / QRadioButton ---- */
        QCheckBox, QRadioButton {{
            font-family: "{Fonts.FAMILY}";
            font-size: {Fonts.SIZE_SMALL}pt;
        }}

        /* ---- QSpinBox / QDoubleSpinBox ---- */
        QSpinBox, QDoubleSpinBox {{
            font-family: "{Fonts.FAMILY}";
            font-size: {Fonts.SIZE_SMALL}pt;
            padding: 2px 4px;
            border: 1px solid {Colors.BORDER_DARK};
            border-radius: 3px;
        }}
        QSpinBox:focus, QDoubleSpinBox:focus {{
            border-color: {Colors.PRIMARY};
        }}

        /* ---- QTextEdit ---- */
        QTextEdit {{
            font-family: "{Fonts.FAMILY}";
            font-size: {Fonts.SIZE_SMALL}pt;
        }}

        /* ---- QScrollArea ---- */
        QScrollArea {{
            border: none;
        }}

        /* ---- QMenuBar ---- */
        QMenuBar {{
            font-family: "{Fonts.FAMILY}";
            font-size: {Fonts.SIZE_BODY}pt;
        }}
        QMenu {{
            font-family: "{Fonts.FAMILY}";
            font-size: {Fonts.SIZE_SMALL}pt;
        }}

        /* ---- QStatusBar ---- */
        QStatusBar {{
            font-family: "{Fonts.FAMILY}";
            font-size: {Fonts.SIZE_SMALL}pt;
        }}
    """
