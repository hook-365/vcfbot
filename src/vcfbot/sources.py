"""Registry of source PDFs to ingest. Expand as we widen the crawl."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    name: str
    url: str        # canonical PDF URL on techdocs
    web_url: str    # landing page on techdocs HTML for the same doc set
    product: str
    version: str


SOURCES: list[Source] = [
    Source(
        name="vmware-cloud-foundation-9-1",
        url="https://techdocs.broadcom.com/content/dam/broadcom/techdocs/us/en/pdf/vmware/vcf/vcf-90/vmware-cloud-foundation-9-1.pdf",
        web_url="https://techdocs.broadcom.com/us/en/vmware-cis/vcf/vcf-9-0-and-later/9-1.html",
        product="VCF",
        version="9.1",
    ),
]


SOURCE_BY_NAME: dict[str, Source] = {s.name: s for s in SOURCES}
