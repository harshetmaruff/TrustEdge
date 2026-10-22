"""Source of truth for the fictional TrustEdge evaluation corpus.

Every company, product, person and figure below is invented for this research
prototype. Nothing here describes a real organisation, product or policy, and
no real personal data is used (see AGENTS.md, "STRICT CONSTRAINTS").

Running this module as a script regenerates:

    data/documents/*.txt    one plain-text document per source
    data/questions.jsonl    the labelled evaluation set

The generated files are committed so that evaluation runs are reproducible
without re-running the generator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

DATA_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = DATA_DIR / "documents"
QUESTIONS_PATH = DATA_DIR / "questions.jsonl"

# --------------------------------------------------------------------------
# Fictional source documents
# --------------------------------------------------------------------------

DOCUMENTS: List[Dict[str, Any]] = [
    {
        "doc_id": "vantis_vt40_manual",
        "title": "Vantis Instruments Ltd. - VT-40 Flow Meter Product Manual",
        "sections": [
            (
                "1. Overview",
                [
                    "The VT-40 Flow Meter is a compact inline sensor manufactured by Vantis Instruments Ltd. "
                    "It measures volumetric flow rate in litres per minute for liquids carried through rigid pipework "
                    "up to 40 millimetres in diameter.",
                    "Each VT-40 ships with the standard 5 metre shielded cable, the Vantis mounting bracket set, and a "
                    "calibration certificate that remains valid for 12 months from the date of shipment.",
                ],
            ),
            (
                "2. Installation",
                [
                    "The VT-40 must be installed in a straight pipe run with at least 20 diameters of upstream clearance "
                    "and 10 diameters of downstream clearance measured from the meter body.",
                    "Installation on a vertical pipe run is permitted only when flow is upward. Downward flow causes the "
                    "internal sensor to settle against the upper housing and produces an unstable baseline reading.",
                    "The unit is rated for operation between 0 and 45 degrees Celsius and at relative humidity up to "
                    "85 percent. It is not rated for outdoor exposure without an additional IP67 weather shroud.",
                ],
            ),
            (
                "3. Calibration",
                [
                    "Factory calibration is performed against a reference meter at the Vantis Marrowby laboratory. The "
                    "supplied certificate records the measured flow error, which is normally within plus or minus 1.5 "
                    "percent across the mid-range of 5 to 25 litres per minute.",
                    "Zero flow calibration must be repeated whenever the meter is moved, or after any replacement of the "
                    "inlet filter. The procedure isolates flow, enters calibration mode with the three-pin service "
                    "jumper, and waits for the on-board indicator to pulse twice.",
                    "Field calibration against a customer-supplied reference requires that reference meter to be within "
                    "0.5 percent of the factory baseline, otherwise the field result is rejected by the firmware.",
                ],
            ),
            (
                "4. Sampling and Output",
                [
                    "The VT-40 samples the flow sensor at 5 Hz and publishes a measurement every 200 milliseconds.",
                    "Each published message contains the timestamp, the flow rate in litres per minute, the internal "
                    "sensor temperature in Celsius, and a two-character status code.",
                    "The status code OK means that the meter is operating within its calibrated range. The status code OV "
                    "indicates that the measured rate exceeded 30 litres per minute. The status code SF indicates a "
                    "sensor fault and should be logged by the host system.",
                    "Communications use a 3.3 volt UART interface at 9600 baud with 8 data bits, one stop bit and no "
                    "parity. A wireless variant, the VT-40W, adds an embedded radio that transmits the same payload at 2 Hz.",
                ],
            ),
            (
                "5. Maintenance and Fault Codes",
                [
                    "Scheduled maintenance consists of inspecting the inlet filter every 6 months and replacing it every "
                    "12 months. Replacing the filter voids the factory calibration certificate and requires recalibration.",
                    "Error code E-11 indicates that the inlet filter is blocked. Error code E-22 indicates that the "
                    "internal temperature sensor has drifted outside its acceptable band. Error codes beginning with E-9 "
                    "indicate a wiring fault on the UART bus.",
                    "The meter retains a rolling event log of the last 128 entries in non-volatile memory. This log "
                    "survives a power cycle but is cleared by a full factory reset.",
                ],
            ),
            (
                "6. Warranty",
                [
                    "Standard warranty is 24 months from the date of delivery. Extended warranty adds 12 months and is "
                    "available only through authorised distributors.",
                    "Warranty does not cover damage caused by reverse flow, by operation outside the stated temperature "
                    "range, or by use of third-party replacement filters.",
                    "Warranty claims for meters installed outdoors without the approved shroud are rejected.",
                ],
            ),
        ],
    },
    {
        "doc_id": "kestrel_freight_policy",
        "title": "Kestrel Freight - Shipping, Insurance and Claims Policy",
        "sections": [
            (
                "1. Service Levels and Zones",
                [
                    "Kestrel Freight operates a regional parcel network covering six domestic zones. Zone A is the "
                    "Kestrel metro area, Zone B covers the adjacent coastal ring, and Zones C through F cover the "
                    "remaining inland corridors. Transit times are quoted in working days and exclude public holidays.",
                ],
            ),
            (
                "2. Transit Times",
                [
                    "Standard service is offered in every zone and is quoted at 3 working days for Zone A and 5 working "
                    "days for Zones B through F.",
                    "Express service is offered only in Zones A, B and C. Express transit is 1 working day in Zone A and "
                    "2 working days in Zones B and C.",
                    "Consignments for Zones D, E and F cannot be sent using Express service.",
                    "Saturday delivery is available in Zone A for an additional surcharge of 4.50 credits. No delivery is "
                    "offered on Sundays or on public holidays.",
                    "Perishable consignments must be handed over for dispatch before 16:00 local time to qualify for "
                    "next-working-day handling.",
                ],
            ),
            (
                "3. Pricing",
                [
                    "The base charge is 3.20 credits for the first item and 1.40 credits for each additional item in the "
                    "same consignment.",
                    "Oversized consignments, meaning any item whose longest side exceeds 90 centimetres, carry a "
                    "surcharge of 6 credits.",
                    "Residential delivery carries a surcharge of 1.10 credits per consignment. Collection from a Kestrel "
                    "service point is free.",
                ],
            ),
            (
                "4. Insurance Cover",
                [
                    "Kestrel insures consignments against accidental loss and damage up to a declared value. The maximum "
                    "declared value for a single consignment is 800 credits.",
                    "The sender must mark the consignment as fragile if the contents are sensitive to impact. "
                    "Fragile-marked consignments that are damaged in transit are replaced rather than reimbursed at the "
                    "declared value.",
                    "Consequential loss, including loss of profit and business interruption, is excluded from cover under "
                    "all service levels.",
                ],
            ),
            (
                "5. Claims",
                [
                    "Claims must be submitted within 14 calendar days of the delivery date. Claims submitted after that "
                    "window are declined.",
                    "Claims are adjudicated by a Kestrel claims assessor within 5 working days of receipt. Where the "
                    "evidence supplied is insufficient to establish the cause of damage, the claim may be referred for "
                    "physical inspection, which adds a further 10 working days.",
                    "Approved claims are settled by credit note against the sender's account. Cash settlement is not "
                    "available for commercial accounts.",
                ],
            ),
            (
                "6. Prohibited Items",
                [
                    "Loose lithium cells, unframed blades and pressurised aerosols are not accepted. Firearms and "
                    "ammunition require a registered firearms dealer account.",
                    "Perishable food items are accepted only when packed with dry ice and declared in advance.",
                    "Livestock, and any consignment containing live animals, is prohibited.",
                ],
            ),
            (
                "7. Accounts and Payment",
                [
                    "Accounts are invoiced monthly in arrears and payment is due within 21 days of the invoice date.",
                    "Commercial accounts must hold a credit limit of at least 500 credits to open an account. The credit "
                    "limit is reviewed every quarter.",
                ],
            ),
        ],
    },
    {
        "doc_id": "lumen_bay_fellowship",
        "title": "Lumen Bay Labs - Research Fellowship Policy",
        "sections": [
            (
                "1. Fellowship Schemes",
                [
                    "Lumen Bay Labs awards an annual fellowship to support doctoral research in applied machine learning. "
                    "The programme is administered by the Lumen Bay research office, which reports to the Lumen Bay "
                    "scientific council.",
                    "The flagship Lumen Bay Doctoral Fellowship carries a stipend of 34,000 credits over three years and "
                    "is awarded to one candidate each year.",
                    "The Lumen Bay Visiting Fellowship is available to researchers from institutions outside the "
                    "Lumen Bay partner list. It funds travel and accommodation for a maximum stay of 4 months and does "
                    "not include a stipend.",
                    "A separate Lumen Bay Microgrant scheme offers awards of up to 5,000 credits for projects with a "
                    "planned duration of no more than 12 months.",
                ],
            ),
            (
                "2. Eligibility",
                [
                    "Candidates for the Doctoral Fellowship must hold a first-class or second-class degree in a relevant "
                    "subject. Candidates must not have been enrolled in a doctoral programme for more than 24 months at "
                    "the time of application.",
                    "Candidates who have already completed a doctoral degree are not eligible to apply for the Doctoral "
                    "Fellowship.",
                    "The fellowship is open to candidates of any nationality, but candidates who require a visa to study "
                    "in Lumen Bay must have been resident in an eligible country for at least 12 months before applying.",
                    "Funding is not awarded to candidates who currently hold a permanent research post elsewhere. A "
                    "fellowship must be the candidate's primary source of research support during the award period.",
                    "Researchers with a prior Lumen Bay microgrant may apply for the Doctoral Fellowship, provided the "
                    "microgrant closed at least 9 months before the application deadline.",
                ],
            ),
            (
                "3. Stipend and Research Expenses",
                [
                    "The Doctoral Fellowship stipend is paid monthly by bank transfer on the 28th of each month. A 12 "
                    "percent contribution is withheld for pension purposes and is paid to the pension provider on the "
                    "fellowship anniversary.",
                    "Research expenses are funded separately from the stipend, up to a total of 7,500 credits per year "
                    "for equipment and 2,000 credits per year for travel and conference fees.",
                    "Funding for a research visit of more than 14 consecutive days requires prior written approval from "
                    "the research office.",
                ],
            ),
            (
                "4. Duration and Extension",
                [
                    "The Doctoral Fellowship runs for three years and may be extended by up to 12 months in exceptional "
                    "circumstances, subject to council approval. Extensions are not granted as a matter of course.",
                    "Visiting Fellowships are limited to a maximum of 4 months in any 24-month period.",
                ],
            ),
            (
                "5. Reporting and Publications",
                [
                    "Fellows must submit a written progress report at the end of each 6-month period and a final report "
                    "within 60 days of the fellowship ending.",
                    "Progress reports are reviewed by the Lumen Bay scientific council, which may request a meeting with "
                    "the fellow and the supervisor.",
                    "Any publication arising from fellowship-funded work must acknowledge the Lumen Bay Fellowship and "
                    "must be deposited in the Lumen Bay repository.",
                ],
            ),
            (
                "6. Conduct and Intellectual Property",
                [
                    "Intellectual property arising from fellowship work belongs to the fellow and to the host "
                    "institution jointly, unless a separate agreement is signed before the fellowship begins.",
                    "Fellows must disclose any outside employment or consulting arrangement to the research office "
                    "within 15 working days of starting, and whenever that arrangement changes.",
                    "Ethics approval is required before any work involving human participants, patient data or clinical "
                    "intervention.",
                ],
            ),
        ],
    },
    {
        "doc_id": "orrin_support_sla",
        "title": "Orrin Retail Group - Customer Support Service Levels",
        "sections": [
            (
                "1. Support Channels",
                [
                    "Orrin Retail Group sells household appliances under the Orrin brand and operates an in-house "
                    "support centre.",
                    "Customer support is available by telephone, email and live chat. Live chat is staffed between 08:00 "
                    "and 20:00 local time, every day except Sunday. Email support is monitored continuously but is not "
                    "staffed on Sunday.",
                    "Telephone support is offered for a fee of 2.00 credits per call for non-warranty enquiries. "
                    "Warranty-related telephone calls are free.",
                ],
            ),
            (
                "2. Response Times",
                [
                    "Live chat is answered within 2 minutes during staffed hours. Email enquiries receive a first response "
                    "within 1 working day.",
                    "Warranty claims receive a decision within 5 working days of receiving the claim. The decision is "
                    "communicated by email.",
                ],
            ),
            (
                "3. Warranty",
                [
                    "Appliances carry a 2-year warranty covering manufacturing faults. Accidental damage and consumable "
                    "parts are not covered.",
                    "The warranty period starts on the delivery date recorded by the courier, not on the order date.",
                    "Extended warranty can be purchased within 60 days of delivery for a fee of 24 credits and adds 3 "
                    "years of cover for parts and labour.",
                ],
            ),
            (
                "4. Returns and Refunds",
                [
                    "Unused appliances in their original packaging can be returned within 30 days of delivery for a full "
                    "refund. The refund is issued to the original payment method within 14 days of the returned item "
                    "being received and inspected.",
                    "Refurbished appliances are sold with a 12-month warranty and may not be returned except where the "
                    "unit is faulty on delivery.",
                    "A 25 percent restocking fee applies to returns of opened appliances that are not faulty, provided "
                    "the item is returned within 30 days.",
                ],
            ),
            (
                "5. Escalation",
                [
                    "Unresolved enquiries are escalated to a senior adviser after 3 separate contacts about the same "
                    "issue. Complaints about the conduct of a support agent are handled separately and are never counted "
                    "towards the 3-contact escalation threshold.",
                    "Claims above 400 credits in value are referred to the Orrin dispute panel, which responds within "
                    "20 working days.",
                ],
            ),
            (
                "6. Feedback",
                [
                    "Customers are asked to rate the support interaction out of 5. Ratings below 3 are reviewed by the "
                    "support quality team within 10 working days.",
                    "Customer satisfaction scores are reported to the Orrin board each quarter.",
                ],
            ),
        ],
    },
]


# --------------------------------------------------------------------------
# Labelled evaluation set
# --------------------------------------------------------------------------
#
# `category` describes the *question type* the dataset was built to cover.
# `gold_label` is the answer the trust layer should produce for that question,
# so edge-case questions are still scorable.
#
# `reference_evidence` is documentation for humans only: no component of the
# pipeline reads it, and it never influences retrieval or scoring.

QUESTIONS: List[Dict[str, str]] = [
    # ---------------- supported ----------------
    {
        "qid": "q01",
        "question": "What is the sampling rate of the VT-40 Flow Meter?",
        "category": "supported",
        "gold_label": "supported",
        "rationale": "Sampling rate is stated explicitly in section 4.",
        "reference_evidence": "The VT-40 samples the flow sensor at 5 Hz and publishes a measurement every 200 milliseconds.",
    },
    {
        "qid": "q02",
        "question": "What transit time does Kestrel Freight quote for standard service to Zone A?",
        "category": "supported",
        "gold_label": "supported",
        "rationale": "Standard transit time for Zone A is stated in section 2.",
        "reference_evidence": "Standard service is offered in every zone and is quoted at 3 working days for Zone A and 5 working days for Zones B through F.",
    },
    {
        "qid": "q03",
        "question": "How long is the standard warranty on Orrin appliances?",
        "category": "supported",
        "gold_label": "supported",
        "rationale": "Standard appliance warranty length is stated in section 3.",
        "reference_evidence": "Appliances carry a 2-year warranty covering manufacturing faults.",
    },
    {
        "qid": "q04",
        "question": "What is the maximum declared value for a single Kestrel Freight consignment?",
        "category": "supported",
        "gold_label": "supported",
        "rationale": "The cap on declared value is stated in section 4.",
        "reference_evidence": "Kestrel insures consignments against accidental loss and damage up to a declared value. The maximum declared value for a single consignment is 800 credits.",
    },
    {
        "qid": "q05",
        "question": "What does the Lumen Bay Visiting Fellowship fund?",
        "category": "supported",
        "gold_label": "supported",
        "rationale": "Funding scope of the visiting scheme is stated in section 1.",
        "reference_evidence": "It funds travel and accommodation for a maximum stay of 4 months and does not include a stipend.",
    },
    {
        "qid": "q06",
        "question": "What is the Kestrel Freight base charge for the first item in a consignment?",
        "category": "supported",
        "gold_label": "supported",
        "rationale": "Base charge is stated in section 3.",
        "reference_evidence": "The base charge is 3.20 credits for the first item and 1.40 credits for each additional item in the same consignment.",
    },
    {
        "qid": "q07",
        "question": "What percentage of the Lumen Bay Doctoral Fellowship stipend is withheld for pension?",
        "category": "supported",
        "gold_label": "supported",
        "rationale": "The pension withholding rate is stated in section 3.",
        "reference_evidence": "A 12 percent contribution is withheld for pension purposes and is paid to the pension provider on the fellowship anniversary.",
    },
    {
        "qid": "q08",
        "question": "How often must Lumen Bay fellows submit progress reports?",
        "category": "supported",
        "gold_label": "supported",
        "rationale": "Reporting cadence is stated in section 5.",
        "reference_evidence": "Fellows must submit a written progress report at the end of each 6-month period and a final report within 60 days of the fellowship ending.",
    },
    {
        "qid": "q09",
        "question": "What does the VT-40 status code SF indicate?",
        "category": "supported",
        "gold_label": "supported",
        "rationale": "Status code meanings are stated in section 4.",
        "reference_evidence": "The status code SF indicates a sensor fault and should be logged by the host system.",
    },
    # ---------------- unsupported ----------------
    {
        "qid": "q10",
        "question": "Kestrel Express delivery is available in every zone. How long does it take to reach Zone F?",
        "category": "unsupported",
        "gold_label": "unsupported",
        "rationale": "False premise: the policy states Express covers only Zones A, B and C.",
        "reference_evidence": "Express service is offered only in Zones A, B and C.",
    },
    {
        "qid": "q11",
        "question": "Are faulty appliances returned to Orrin subject to a 25 percent restocking fee?",
        "category": "unsupported",
        "gold_label": "unsupported",
        "rationale": "Contradicts the policy, which applies the restocking fee to non-faulty returns.",
        "reference_evidence": "A 25 percent restocking fee applies to returns of opened appliances that are not faulty, provided the item is returned within 30 days.",
    },
    {
        "qid": "q12",
        "question": "Must Lumen Bay Doctoral Fellowship candidates have already completed a PhD?",
        "category": "unsupported",
        "gold_label": "unsupported",
        "rationale": "Contradicts the eligibility rule for candidates still within 24 months of enrolment.",
        "reference_evidence": "Candidates must not have been enrolled in a doctoral programme for more than 24 months at the time of application.",
    },
    {
        "qid": "q13",
        "question": "A Lumen Bay fellow may travel for up to 21 consecutive days without seeking approval.",
        "category": "unsupported",
        "gold_label": "unsupported",
        "rationale": "Contradicts the 14-day approval threshold.",
        "reference_evidence": "Funding for a research visit of more than 14 consecutive days requires prior written approval from the research office.",
    },
    {
        "qid": "q14",
        "question": "Does the VT-40 Flow Meter support wireless transmission at 5 Hz?",
        "category": "unsupported",
        "gold_label": "unsupported",
        "rationale": "Contradicts the spec: the base unit is wired UART, and the wireless variant transmits at 2 Hz.",
        "reference_evidence": "Communications use a 3.3 volt UART interface at 9600 baud... A wireless variant, the VT-40W, adds an embedded radio that transmits the same payload at 2 Hz.",
    },
    {
        "qid": "q15",
        "question": "Can a Kestrel Freight customer ship livestock if it is declared in advance?",
        "category": "unsupported",
        "gold_label": "unsupported",
        "rationale": "Contradicts the outright prohibition on livestock.",
        "reference_evidence": "Livestock, and any consignment containing live animals, is prohibited.",
    },
    # ---------------- abstain ----------------
    {
        "qid": "q16",
        "question": "How many items can be included in a single Kestrel Freight consignment?",
        "category": "abstain",
        "gold_label": "abstain",
        "rationale": "Pricing states a per-item rate but no maximum item count.",
        "reference_evidence": "",
    },
    {
        "qid": "q17",
        "question": "Which courier service does Orrin Retail Group use for deliveries?",
        "category": "abstain",
        "gold_label": "abstain",
        "rationale": "The courier is never named in the policy.",
        "reference_evidence": "",
    },
    {
        "qid": "q18",
        "question": "How many Lumen Bay fellows may hold a Microgrant in the same year?",
        "category": "abstain",
        "gold_label": "abstain",
        "rationale": "The microgrant scheme has a value cap and duration cap but no quota.",
        "reference_evidence": "",
    },
    {
        "qid": "q19",
        "question": "What is the flow accuracy of the VT-40 above 30 litres per minute?",
        "category": "abstain",
        "gold_label": "abstain",
        "rationale": "Accuracy is only guaranteed across 5 to 25 litres per minute; beyond 30 the status code reports overflow.",
        "reference_evidence": "",
    },
    {
        "qid": "q20",
        "question": "How many staff does the Orrin support centre employ?",
        "category": "abstain",
        "gold_label": "abstain",
        "rationale": "The policy describes channels and staffing hours but never gives a headcount.",
        "reference_evidence": "",
    },
    # ---------------- edge case ----------------
    {
        "qid": "q21",
        "question": "How long do I have to file a damage claim with Kestrel Freight?",
        "category": "edge_case",
        "gold_label": "supported",
        "rationale": "The claims window sits in a different section from the insurance cover, so the right chunk must be found before it can be answered.",
        "reference_evidence": "Claims must be submitted within 14 calendar days of the delivery date.",
    },
    {
        "qid": "q22",
        "question": "If I return an unopened Orrin appliance after 20 days, do I get a full refund and when is the refund paid?",
        "category": "edge_case",
        "gold_label": "supported",
        "rationale": "Requires combining the 30-day return window with the 14-day refund settlement rule.",
        "reference_evidence": "Unused appliances in their original packaging can be returned within 30 days of delivery for a full refund. The refund is issued to the original payment method within 14 days of the returned item being received and inspected.",
    },
    {
        "qid": "q23",
        "question": "Can I send lithium batteries to Kestrel if I pack them in a padded case?",
        "category": "edge_case",
        "gold_label": "abstain",
        "rationale": "The policy bans 'loose lithium cells' but never defines 'loose', so the padded-case case cannot be resolved.",
        "reference_evidence": "",
    },
    {
        "qid": "q24",
        "question": "What is the maximum ambient temperature the VT-40W wireless variant is rated for?",
        "category": "edge_case",
        "gold_label": "abstain",
        "rationale": "The temperature rating is given for the VT-40 only; the VT-40W is described solely as adding a radio, so the variant's rating is never stated.",
        "reference_evidence": "",
    },
    {
        "qid": "q25",
        "question": "Which Lumen Bay fellowship scheme provides the highest value per month to the recipient?",
        "category": "edge_case",
        "gold_label": "abstain",
        "rationale": "Comparing the three schemes requires a value basis the policy never defines, and the visiting scheme has no stipend at all.",
        "reference_evidence": "",
    },
]


def render_document(document: Dict[str, Any]) -> str:
    """Render one document dict into the plain-text format stored on disk.

    The format uses markdown-style headers so that section titles survive into
    the chunks used for retrieval and can be shown as evidence context.
    """
    lines: List[str] = [f"# {document['title']}", ""]
    for section_title, paragraphs in document["sections"]:
        lines.append(f"## {section_title}")
        lines.append("")
        for paragraph in paragraphs:
            lines.append(paragraph)
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_documents(documents_dir: Path = DOCUMENTS_DIR) -> List[Path]:
    """Write every document to disk and return the written paths."""
    documents_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for document in DOCUMENTS:
        path = documents_dir / f"{document['doc_id']}.txt"
        path.write_text(render_document(document), encoding="utf-8")
        written.append(path)
    return written


def write_questions(path: Path = QUESTIONS_PATH) -> Path:
    """Write the labelled question set as JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for question in QUESTIONS:
            handle.write(json.dumps(question, ensure_ascii=False) + "\n")
    return path


def generate_all(documents_dir: Path = DOCUMENTS_DIR, questions_path: Path = QUESTIONS_PATH) -> Tuple[List[Path], Path]:
    """Regenerate both the document corpus and the labelled question set."""
    return write_documents(documents_dir), write_questions(questions_path)


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    docs, questions_file = generate_all()
    for doc in docs:
        print(f"wrote {doc}")
    print(f"wrote {questions_file}")
