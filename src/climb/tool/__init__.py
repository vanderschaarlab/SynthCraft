from typing import Any, Dict, List, Optional

from climb.common.utils import check_extra_available

from .impl.tool_autoprognosis_explainers import AutoprognosisExplainerInvase, AutoprognosisExplainerSymbolicPursuit 
from .impl.tool_autoprognosis import (
    AutoprognosisClassification,
    AutoprognosisClassificationTrainTest,
    AutoprognosisRegression,
    AutoprognosisRegressionTrainTest,
    AutoprognosisSubgroupEvaluation,
    AutoprognosisSurvival,
    AutoprognosisSurvivalTrainTest,
)
from .impl.tool_balance_data import BalanceData
from .impl.tool_data_centric import DataIQInsights
from .impl.tool_data_suite import DataSuiteInsights
from .impl.tool_descriptive_stats import DescriptiveStatistics
from .impl.tool_exploratory_data_analysis import ExploratoryDataAnalysis
from .impl.tool_feature_extraction_from_text import FeatureExtractionFromText
from .impl.tool_feature_importance import PermutationExplainer, ShapExplainer
from .impl.tool_feature_selection import BorutaFeatureSelection
from .impl.tool_hardware import HardwareInfo
from .impl.tool_imputation import HyperImputeImputation, HyperImputeImputationTrainTest
from .impl.tool_paper import UploadAndSummarizeExamplePaper
from .impl.tool_smart_testing import SmartTesting
from .impl.tool_upload import UploadDataFile, UploadDataMultipleFiles
from .tool_comms import ToolCommunicator, ToolOutput, ToolReturnIter
from .tools import ToolBase, UserInputRequest

if check_extra_available():
    # Any tools that are incompatible with Apache 2.0 license should be imported here.
    from .impl_agpl.tool_data_valuation import KNNShapleyValuation
    from .impl_agpl.tool_outlier_detection import CleanlabOutlierDetection


def get_tool(tool_name: str) -> ToolBase:
    return AVAILABLE_TOOLS[tool_name]


def list_all_tool_specs(filter_tool_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    tool_specs = [
        tool.specification
        for tool in _AVAILABLE_TOOLS_LIST
        if (filter_tool_names is None or tool.name in filter_tool_names)
    ]

    # Validate tool specs:
    for tool in tool_specs:
        if "function" not in tool:
            raise ValueError(f"Tool spec must have a 'function' key: {tool}")
        func = tool["function"]
        if "name" not in func:
            raise ValueError(f"Tool spec's function must have a 'name' key: {tool}")
        if "description" not in func:
            raise ValueError(f"Tool spec's function must have a 'description' key: {tool}")
        if len(func["description"]) > 1027:
            raise ValueError(f"Tool spec's function description must be 1027 characters or less: {tool}")
        # NOTE: Not meant to be extensive validation.

    return tool_specs


def list_all_tool_names(filter_tool_names: Optional[List[str]] = None) -> List[str]:
    names = [
        tool.name for tool in _AVAILABLE_TOOLS_LIST if (filter_tool_names is None or tool.name in filter_tool_names)
    ]
    return names


_AVAILABLE_TOOLS_LIST = [
    AutoprognosisExplainerInvase(),
    AutoprognosisExplainerSymbolicPursuit(),
    AutoprognosisClassification(),
    AutoprognosisRegression(),
    AutoprognosisSubgroupEvaluation(),
    AutoprognosisSurvival(),
    AutoprognosisClassificationTrainTest(),
    AutoprognosisRegressionTrainTest(),
    AutoprognosisSurvivalTrainTest(),
    BorutaFeatureSelection(),
    DataIQInsights(),
    DescriptiveStatistics(),
    ExploratoryDataAnalysis(),
    HardwareInfo(),
    HyperImputeImputation(),
    HyperImputeImputationTrainTest(),
    PermutationExplainer(),
    ShapExplainer(),
    UploadAndSummarizeExamplePaper(),
    UploadDataFile(),
    UploadDataMultipleFiles(),
    DataSuiteInsights(),
    BalanceData(),
    FeatureExtractionFromText(),
    SmartTesting(),
]

if check_extra_available():
    # Any tools that are incompatible with Apache 2.0 license should added here.
    _AVAILABLE_TOOLS_LIST.extend(
        [
            CleanlabOutlierDetection(),
            KNNShapleyValuation(),
        ]
    )

AVAILABLE_TOOLS = {tool.name: tool for tool in _AVAILABLE_TOOLS_LIST}

__all__ = [
    "get_tool",
    "list_all_tool_specs",
    "ToolBase",
    "ToolCommunicator",
    "ToolOutput",
    "ToolReturnIter",
    "UserInputRequest",
]
