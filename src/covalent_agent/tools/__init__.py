"""MCP tool definitions for agent capabilities."""

from .chemistry_tools import TOOLS as CHEMISTRY_TOOLS
from .literature_tools import TOOLS as LITERATURE_TOOLS
from .protein_tools import TOOLS as PROTEIN_TOOLS

ALL_TOOLS = PROTEIN_TOOLS + CHEMISTRY_TOOLS + LITERATURE_TOOLS
