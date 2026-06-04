"""VCF sizing calculator — drives the Planning & Preparation Workbook's OWN
formulas as the engine.

We do NOT re-implement Broadcom's sizing math (host capacity, vSAN overhead,
growth reserves, HA multipliers). Instead we load the workbook, set the input
cells, let its 8,990 formulas recalculate, and read the output cells. When
Broadcom republishes the workbook for a new version, `vcfbot fetch` pulls the
new xlsx and the logic updates itself — no reverse-engineering, no drift.

Compiling the formula graph takes ~60s; we do it once, lazily, and cache the
ExcelModel as a module singleton. Each subsequent `compute()` is a ~4s recalc.

Input/output cells were mapped from the "Management Domain Sizing" sheet of the
VCF 9.1 workbook (col E holds the editable knobs; cols G-M hold the computed
component table; row 33 holds the totals).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

from .config import load_settings
from .sources import vcf_workbook

_SHEET = "MANAGEMENT DOMAIN SIZING"

# Friendly input name -> cell on the Management Domain Sizing sheet (col E).
INPUT_CELLS: dict[str, str] = {
    "host_ops_reserve_pct": "E8",     # Host and Operations Reserve (%)
    "storage_growth_pct": "E9",       # Storage Estimated Growth (%)
    "host_cpu_cores": "E13",          # CPU Cores per host
    "host_ram_gb": "E14",             # RAM per host (GB)
    "cpu_oversubscription": "E15",    # CPU Oversubscription (X:1)
    "memory_oversubscription": "E16",  # Memory Oversubscription (X:1)
    "instance_model": "E20",          # First Instance | Additional Instance | VVF
    "availability_model": "E21",      # High Availability | Standard
    "size": "E22",                    # management vCenter size profile
    # Per-component Include/Exclude toggles.
    "log_management": "E25",
    "log_management_replicas": "E26",
    "vcf_operations_for_networks": "E27",
    "realtime_metrics": "E28",
    "software_depot": "E29",
    "identity_broker": "E30",
    "vcf_operations": "E32",
    "vcf_automation": "E33",
    "cloud_proxy": "E34",
}

# Component output table: label in col G, [nodes, vCPU, RAM, disk] in J/K/L/M.
_LABEL_COL = "G"
_OUT_COLS = {"nodes": "J", "vcpu": "K", "ram_gb": "L", "disk_gb": "M"}
_COMPONENT_ROWS = range(8, 33)   # management + workload component rows
_TOTALS_ROW = 33


# Valid option lists + friendly labels for the UI form. Values are the literal
# strings the workbook's dropdowns expect.
# NOTE: "Tiny" is omitted on purpose — it is not a supported management-domain
# vCenter size, and selecting it makes the workbook NULL the totals row (same
# failure mode as the invalid "Standard" availability value). Small is the floor.
SIZE_OPTIONS = ["Small", "Medium", "Large", "XLarge"]
AVAILABILITY_OPTIONS = ["Simple", "High Availability"]
INSTANCE_OPTIONS = ["First Instance", "Additional Instance", "VVF"]

# Components that are REQUIRED for a supported deployment, not truly optional.
# The workbook lets you toggle them off for modeling, but the docs make VCF
# Operations mandatory ("VCF Operation is the mandatory requirement along with
# vCenter"), so the planner deploys it by default and flags it required. Toggling
# it on also auto-adds the License Server (the workbook's own dependency logic).
REQUIRED_COMPONENTS = {"vcf_operations"}
# Optional components (the Include/Exclude toggles), with display labels.
COMPONENT_LABELS = {
    "vcf_operations": "VCF Operations",
    "vcf_automation": "VCF Automation",
    "vcf_operations_for_networks": "VCF Operations for Networks",
    "realtime_metrics": "Real-time Metrics",
    "log_management": "Log Management",
    "log_management_replicas": "Log Management Replicas",
    "identity_broker": "Identity Broker (additional)",
    "software_depot": "Software Depot (additional)",
    "cloud_proxy": "Cloud Proxy",
}

# Sensible "smallest SUPPORTED" starting point: smallest valid sizes, no HA,
# truly-optional components off, but REQUIRED components (VCF Operations) on so
# the baseline is actually a supported deployment. The user scales up from here.
DEFAULTS = {
    "size": "Small",
    "availability_model": "Simple",
    "instance_model": "First Instance",
    "host_cpu_cores": 64,
    "host_ram_gb": 512,
    "cpu_oversubscription": 1,
    "memory_oversubscription": 1,
    "host_ops_reserve_pct": 30,
    "storage_growth_pct": 10,
    "vcf_operations": "Include",  # mandatory — see REQUIRED_COMPONENTS
}


def options() -> dict:
    """Form metadata for the Planner UI: valid choices, components, defaults."""
    return {
        "sizes": SIZE_OPTIONS,
        "availability": AVAILABILITY_OPTIONS,
        "instance_models": INSTANCE_OPTIONS,
        "components": [
            {"key": k, "label": v, "required": k in REQUIRED_COMPONENTS}
            for k, v in COMPONENT_LABELS.items()
        ],
        "defaults": DEFAULTS,
    }


@dataclass
class ComponentSize:
    name: str
    nodes: float
    vcpu: float
    ram_gb: float
    disk_gb: float


@dataclass
class SizingResult:
    components: list[ComponentSize] = field(default_factory=list)
    total_nodes: float = 0.0
    total_vcpu: float = 0.0
    total_ram_gb: float = 0.0
    total_disk_gb: float = 0.0
    inputs: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "components": [
                {
                    "name": c.name,
                    "nodes": c.nodes,
                    "vcpu": c.vcpu,
                    "ram_gb": c.ram_gb,
                    "disk_gb": c.disk_gb,
                }
                for c in self.components
            ],
            "totals": {
                "nodes": self.total_nodes,
                "vcpu": self.total_vcpu,
                "ram_gb": self.total_ram_gb,
                "disk_gb": self.total_disk_gb,
            },
            "inputs": self.inputs,
        }


# Module singleton: the compiled formula model + the resolved full-key prefix.
_model = None
_key_for: dict[str, str] = {}
_lock = Lock()


def _num(v):
    """Coerce a workbook cell value (np scalar / str like '114 vCPUs') to float."""
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"-?\d+(?:\.\d+)?", str(v))
    return float(m.group(0)) if m else 0.0


def _workbook_path() -> Path:
    settings = load_settings()
    return settings.pdf_dir / f"{vcf_workbook(settings_version()).name}.xlsx"


def settings_version() -> str:
    # The single source-of-truth version lives on the workbook Source.
    return "9.1"


def _load_model():
    """Compile the workbook once (~60s) and cache it + a cell->full-key map."""
    global _model, _key_for
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        import formulas  # heavy; imported lazily so the rest of the app is unaffected

        path = _workbook_path()
        if not path.exists():
            raise FileNotFoundError(
                f"workbook not found at {path} — run `vcfbot fetch` first"
            )
        model = formulas.ExcelModel().loads(str(path)).finish()
        sol = model.calculate()
        # Resolve the engine's full cell keys (they carry a [BOOK]SHEET! prefix)
        # so callers can address cells as plain refs like "E22" / "K33".
        for k in sol:
            up = k.upper()
            if _SHEET in up and "!" in up:
                _key_for[up.split("!")[-1]] = k
        _model = model
        return _model


def warm() -> None:
    """Kick off the ~60s compile in a background thread so the first real
    request doesn't block. Safe to call at server startup; no-op if already
    compiled or if the workbook isn't present yet.
    """
    import threading

    def _bg():
        try:
            _load_model()
        except Exception:
            pass  # workbook missing / engine error — first request will surface it

    threading.Thread(target=_bg, name="planner-warm", daemon=True).start()


def is_ready() -> bool:
    return _model is not None


def compute(inputs: dict | None = None) -> SizingResult:
    """Recalculate the workbook with the given input overrides and read the
    component sizing table + totals. `inputs` keys are INPUT_CELLS names.
    """
    model = _load_model()
    overrides = {}
    for name, value in (inputs or {}).items():
        cell = INPUT_CELLS.get(name)
        if not cell:
            continue
        full = _key_for.get(cell)
        if full:
            overrides[full] = value

    sol = model.calculate(inputs=overrides) if overrides else model.calculate()

    def val(cell: str):
        full = _key_for.get(cell)
        if not full or full not in sol:
            return None
        cv = sol[full].value
        return cv[0, 0] if hasattr(cv, "shape") else cv

    result = SizingResult(inputs=dict(inputs or {}))
    for row in _COMPONENT_ROWS:
        label = val(f"{_LABEL_COL}{row}")
        if not isinstance(label, str) or not label.strip():
            continue
        vcpu = _num(val(f"{_OUT_COLS['vcpu']}{row}"))
        ram = _num(val(f"{_OUT_COLS['ram_gb']}{row}"))
        disk = _num(val(f"{_OUT_COLS['disk_gb']}{row}"))
        nodes = _num(val(f"{_OUT_COLS['nodes']}{row}"))
        if vcpu <= 0 and ram <= 0 and disk <= 0:
            continue  # excluded / not-deployed component
        result.components.append(
            ComponentSize(label.strip(), nodes, vcpu, ram, disk)
        )

    result.total_nodes = _num(val(f"{_OUT_COLS['nodes']}{_TOTALS_ROW}"))
    result.total_vcpu = _num(val(f"{_OUT_COLS['vcpu']}{_TOTALS_ROW}"))
    result.total_ram_gb = _num(val(f"{_OUT_COLS['ram_gb']}{_TOTALS_ROW}"))
    result.total_disk_gb = _num(val(f"{_OUT_COLS['disk_gb']}{_TOTALS_ROW}"))
    return result
