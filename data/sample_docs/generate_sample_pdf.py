"""Script to generate a multi-page sample PDF for docqa-rag testing and evaluation."""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def generate_sample_pdf(output_path: str = "data/sample_docs/sample_policy.pdf") -> str:
    """Generate a 4-page realistic technical policy document with distinct facts on each page."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1e3a8a"),
        spaceAfter=14,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=12,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#334155"),
        spaceAfter=8,
    )

    story = []

    # ================= PAGE 1 =================
    story.append(Paragraph("Acme Cloud Platform Operations & Architecture Manual", title_style))
    story.append(Paragraph("Document Version: 2.4 | Classification: Public Engineering Standard", body_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("1. System Architecture & Infrastructure Overview", heading_style))
    story.append(
        Paragraph(
            "Acme Cloud operates distributed Kubernetes clusters running version 1.29 across three primary cloud "
            "regions: us-east (Northern Virginia), us-west (Oregon), and eu-central (Frankfurt). "
            "All production compute workloads are partitioned using namespace isolation and network policies.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "The primary transactional database tier is deployed on PostgreSQL 16 managed via Patroni for automated "
            "high availability clustering. In the event of primary database node degradation or network partition, "
            "automated failover promotes a synchronous replica within a guaranteed window of 15 seconds.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Distributed session caching and global rate limiting are coordinated via a Redis 7.2 cluster topology. "
            "Asynchronous message streaming and audit event ingestion utilize an Apache Kafka 3.6 event bus deployed "
            "with a triple replication factor across independent availability zones.",
            body_style,
        )
    )

    story.append(Paragraph("2. Network Topology & Cryptographic Controls", heading_style))
    story.append(
        Paragraph(
            "All network ingress and egress traffic is enforced through TLS 1.3 protocol encryption. Legacy cipher suites "
            "such as TLS 1.0, 1.1, and 1.2 are rejected by ingress edge load balancers. Inter-service microservice communication "
            "is secured using mutual TLS (mTLS) with certificates rotated automatically every 30 days via cert-manager.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Data at rest across all storage volumes, persistent block stores, and object storage buckets is encrypted "
            "using AES-256 envelope encryption. Encryption keys are managed through customer-managed encryption keys (CMEK) "
            "backed by FIPS 140-2 Level 3 Hardware Security Modules.",
            body_style,
        )
    )

    # ================= PAGE 2 =================
    story.append(PageBreak())
    story.append(Paragraph("3. Identity, Access Management & Authentication", heading_style))
    story.append(
        Paragraph(
            "Acme Cloud enforces strict Role-Based Access Control (RBAC) governed by four canonical system roles: "
            "Admin, Operator, Auditor, and Viewer. All user accounts must authenticate using hardware-backed FIDO2 WebAuthn "
            "security tokens or time-based one-time password (TOTP) multi-factor authentication.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Session tokens for active web and CLI sessions enforce a maximum lifespan of 60 minutes. Cryptographic refresh "
            "tokens are strictly limited to a duration of 24 hours and are revoked upon logout, password change, or IP address shift.",
            body_style,
        )
    )

    story.append(Paragraph("4. Data Retention, Compliance & Purge Schedules", heading_style))
    story.append(
        Paragraph(
            "Compliance guidelines dictate specific retention windows across distinct log categories. System audit trails and "
            "administrative event logs are retained for exactly 365 days within immutable, write-once-read-many (WORM) storage. "
            "Standard application debug logs and tracing metrics are permanently deleted after 90 days.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Right to Erasure requests submitted under GDPR Article 17 or CCPA regulations are scheduled and executed within "
            "a maximum turnaround of 30 calendar days. Upon execution, all associated user data records and cache entries "
            "are completely expunged across primary stores and secondary backup snapshots.",
            body_style,
        )
    )

    # ================= PAGE 3 =================
    story.append(PageBreak())
    story.append(Paragraph("5. Service Level Agreements (SLAs) & Availability Commitments", heading_style))
    story.append(
        Paragraph(
            "Acme Cloud guarantees a 99.95% monthly uptime service level agreement for all active Enterprise Tier subscriptions. "
            "Uptime calculations exclude scheduled maintenance windows communicated at least 7 calendar days in advance.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Service credit compensations are structured on a graduated scale: if monthly uptime falls between 99.0% and 99.95%, "
            "customers are entitled to a 15% billing credit applied to their subsequent invoice. If monthly uptime falls below 99.0%, "
            "customers receive a 30% billing credit upon formal submission of an SLA reimbursement claim within 30 days.",
            body_style,
        )
    )

    story.append(Paragraph("6. Incident Severity Classification & Escalation Matrix", heading_style))
    story.append(
        Paragraph(
            "Incidents are categorized into three operational severity tiers: "
            "Severity 1 (Critical Incident) denotes complete outage or catastrophic data loss affecting multiple enterprise customers. "
            "Initial engineering response time is guaranteed under 15 minutes, with mandatory executive escalation to the Vice President "
            "of Engineering required within 30 minutes of incident declaration.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Severity 2 (Major Incident) designates degradation of core workflows where a viable operational workaround exists. "
            "Initial response SLA is capped at 60 minutes. "
            "Severity 3 (Minor Incident) represents cosmetic interface flaws or non-critical administrative latency, which carries "
            "a resolution SLA within 1 business day.",
            body_style,
        )
    )

    # ================= PAGE 4 =================
    story.append(PageBreak())
    story.append(Paragraph("7. Subscription Pricing Tiers & Resource Quotas", heading_style))
    story.append(
        Paragraph(
            "Acme Cloud provides three standard subscription tiers tailored to organizational size: "
            "Starter Tier is billed at $49 per month and includes access for up to 5 users along with 50 GB of encrypted cloud storage. "
            "Professional Tier is billed at $199 per month, supporting up to 25 users and 500 GB of high-speed NVMe storage.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Enterprise Tier starts at $999 per month. Enterprise accounts include unlimited seat licenses, dedicated multi-region "
            "VPC peering, customized compliance reports, and 24/7 dedicated telephone support with designated technical account managers.",
            body_style,
        )
    )

    story.append(Paragraph("8. Disaster Recovery & Business Continuity", heading_style))
    story.append(
        Paragraph(
            "Disaster recovery protocols mandate strict objective metrics: Recovery Point Objective (RPO) is certified at "
            "5 minutes for stateful transactional data through continuous write-ahead log (WAL) archiving. "
            "Recovery Time Objective (RTO) is certified at 30 minutes for comprehensive multi-region failover operations.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Simulated disaster recovery drills and unannounced failover exercises are executed semi-annually in May and November. "
            "Audit reports of drill results are made available to enterprise compliance officers within 14 days of completion.",
            body_style,
        )
    )

    doc.build(story)
    print(f"Sample PDF successfully generated at: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_sample_pdf()

