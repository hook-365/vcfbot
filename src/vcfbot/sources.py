"""Registry of source PDFs to ingest. Expand as we widen the crawl."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    name: str
    url: str        # canonical download URL on techdocs (PDF or xlsx asset)
    web_url: str    # landing page on techdocs HTML for the same doc set
    product: str
    version: str
    kind: str = "pdf"  # "pdf" (prose doc set) | "xlsx" (structured workbook)


# techdocs URLs embed the version several times, so we template them from a
# SINGLE `version` string rather than hand-writing each (which silently rots on
# a version bump). The two `series_*` slugs are the major-series umbrella that
# Broadcom keeps stable across minors (9.0, 9.1, … all live under "vcf-90" /
# "vcf-9-0-and-later"); they only change on a new major. So a minor bump is one
# edit — `vcf_source("9.2")` — and a major bump overrides the two series slugs.
def vcf_source(
    version: str,
    series_dir: str = "vcf-90",
    series_umbrella: str = "vcf-9-0-and-later",
) -> Source:
    v = version.replace(".", "-")  # "9.1" -> "9-1"
    return Source(
        name=f"vmware-cloud-foundation-{v}",
        url=(
            "https://techdocs.broadcom.com/content/dam/broadcom/techdocs/us/en/"
            f"pdf/vmware/vcf/{series_dir}/vmware-cloud-foundation-{v}.pdf"
        ),
        web_url=(
            "https://techdocs.broadcom.com/us/en/vmware-cis/vcf/"
            f"{series_umbrella}/{v}.html"
        ),
        product="VCF",
        version=version,
    )


def vcf_workbook(version: str = "9.1") -> Source:
    """The VCF Planning & Preparation Workbook (.xlsx).

    The PDF doc set enumerates the management appliances but defers ALL
    per-appliance sizing to this workbook (referenced 26× in the corpus). Its
    Static Reference Tables / Management Domain Sizing sheets carry the actual
    vCPU / RAM / disk per appliance (SDDC Manager, vCenter, NSX Manager, NSX
    Edge, services runtime, …) — the numbers absent from the PDF. The asset URL
    embeds the dotted version verbatim (`vcf-9.1-…`, not `vcf-9-1-…`).
    """
    return Source(
        name=f"vcf-{version}-planning-and-preparation-workbook",
        url=(
            "https://techdocs.broadcom.com/content/dam/broadcom/techdocs/us/en/"
            f"assets/vmware-cis/vcf/vcf-{version}-planning-and-preparation-workbook.xlsx"
        ),
        web_url=(
            "https://techdocs.broadcom.com/us/en/vmware-cis/vcf/"
            "vcf-9-0-and-later/9-1/planning-and-preparation.html"
        ),
        product="VCF",
        version=version,
        kind="xlsx",
    )


SOURCES: list[Source] = [
    vcf_source("9.1"),
    vcf_workbook("9.1"),
]


SOURCE_BY_NAME: dict[str, Source] = {s.name: s for s in SOURCES}
