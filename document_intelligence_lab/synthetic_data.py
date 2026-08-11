from typing import List

from .models import DecisionLabel, Document, DocumentType, EvalQuestion


CASES = [
    {
        "id": "1001",
        "property": "Harbor Lofts",
        "cause": "sudden pipe burst",
        "damage": "water damage to flooring and cabinets",
        "policy": "covered",
        "exclusion": "gradual seepage",
        "inspection": "fresh water lines ruptured suddenly; no long-term staining",
        "amount": "$18,400",
    },
    {
        "id": "1002",
        "property": "Maple Court",
        "cause": "wear and tear roof leak",
        "damage": "ceiling stains and warped drywall",
        "policy": "not_covered",
        "exclusion": "wear and tear or deferred maintenance",
        "inspection": "old roof membrane cracking and repeated patching",
        "amount": "$9,700",
    },
    {
        "id": "1003",
        "property": "Cedar Plaza",
        "cause": "tenant negligence kitchen fire",
        "damage": "smoke damage and appliance replacement",
        "policy": "covered",
        "exclusion": "intentional acts",
        "inspection": "accidental unattended pan fire; no evidence of intent",
        "amount": "$26,200",
    },
    {
        "id": "1004",
        "property": "Riverside Market",
        "cause": "flood from river overflow",
        "damage": "inventory and wall damage",
        "policy": "not_covered",
        "exclusion": "flood or surface water",
        "inspection": "river exceeded banks and entered the loading bay",
        "amount": "$41,900",
    },
    {
        "id": "1005",
        "property": "Northline Offices",
        "cause": "windstorm window breakage",
        "damage": "broken windows and rain intrusion",
        "policy": "covered",
        "exclusion": "open window rain damage",
        "inspection": "windborne debris fractured sealed exterior glazing",
        "amount": "$13,250",
    },
    {
        "id": "1006",
        "property": "Oak Storage",
        "cause": "mold after unresolved humidity",
        "damage": "mold remediation",
        "policy": "not_covered",
        "exclusion": "mold from unresolved humidity",
        "inspection": "humidity logs showed persistent moisture for six months",
        "amount": "$15,600",
    },
    {
        "id": "1007",
        "property": "Pine Apartments",
        "cause": "vandalism",
        "damage": "door and lobby camera damage",
        "policy": "covered",
        "exclusion": "vacancy over 60 days",
        "inspection": "building was occupied; forced entry occurred overnight",
        "amount": "$7,850",
    },
    {
        "id": "1008",
        "property": "Summit Dental",
        "cause": "equipment breakdown",
        "damage": "compressor failure and spoiled supplies",
        "policy": "needs_review",
        "exclusion": "mechanical breakdown unless endorsement applies",
        "inspection": "compressor seized; endorsement status not present in file",
        "amount": "$22,100",
    },
    {
        "id": "1009",
        "property": "Greenfield Leasehold",
        "cause": "leasehold improvement dispute",
        "damage": "custom fixtures after sprinkler discharge",
        "policy": "needs_review",
        "exclusion": "tenant improvements not scheduled",
        "inspection": "fixtures were tenant-installed; schedule page missing",
        "amount": "$31,300",
    },
    {
        "id": "1010",
        "property": "Union Warehouse",
        "cause": "theft",
        "damage": "stolen electronics inventory",
        "policy": "covered",
        "exclusion": "employee dishonesty",
        "inspection": "forced lock damage and police report identify unknown third party",
        "amount": "$38,750",
    },
    {
        "id": "1011",
        "property": "Brighton Clinic",
        "cause": "sewer backup",
        "damage": "flooring and sanitation work",
        "policy": "covered",
        "exclusion": "sewer backup without sublimit",
        "inspection": "backup originated from municipal sewer; policy includes sewer backup sublimit",
        "amount": "$11,950",
    },
    {
        "id": "1012",
        "property": "Aster Retail",
        "cause": "earth movement",
        "damage": "foundation cracking",
        "policy": "not_covered",
        "exclusion": "earth movement",
        "inspection": "settlement and soil movement caused foundation distress",
        "amount": "$58,000",
    },
]


def load_documents() -> List[Document]:
    docs: List[Document] = []
    for case in CASES:
        cid = case["id"]
        label = case["policy"]
        coverage_sentence = {
            "covered": f"The policy covers direct physical loss caused by {case['cause']} when supported by timely notice and repair invoices.",
            "not_covered": f"The policy excludes loss caused by {case['exclusion']}.",
            "needs_review": f"Coverage for {case['cause']} requires an endorsement or schedule page that is not always included in the base policy.",
        }[label]
        docs.extend(
            [
                Document(
                    doc_id=f"POL-{cid}",
                    title=f"Policy excerpt for {case['property']}",
                    doc_type=DocumentType.POLICY,
                    text=(
                        f"Policy number POL-{cid}. Named insured: {case['property']}. "
                        f"{coverage_sentence} The insured must provide invoices, inspection findings, and claim notice."
                    ),
                    metadata={"claim_id": f"CLM-{cid}", "property": case["property"]},
                ),
                Document(
                    doc_id=f"CLM-{cid}",
                    title=f"Claim notice for {case['property']}",
                    doc_type=DocumentType.CLAIM,
                    text=(
                        f"Claim CLM-{cid} reports {case['damage']} at {case['property']}. "
                        f"The reported cause is {case['cause']}. Requested amount is {case['amount']}."
                    ),
                    metadata={"claim_id": f"CLM-{cid}", "property": case["property"]},
                ),
                Document(
                    doc_id=f"INS-{cid}",
                    title=f"Inspection report for {case['property']}",
                    doc_type=DocumentType.INSPECTION,
                    text=(
                        f"Inspection for CLM-{cid}: {case['inspection']}. "
                        f"The observed damage is consistent with {case['cause']}."
                    ),
                    metadata={"claim_id": f"CLM-{cid}", "property": case["property"]},
                ),
                Document(
                    doc_id=f"INV-{cid}",
                    title=f"Invoice packet for {case['property']}",
                    doc_type=DocumentType.INVOICE,
                    text=(
                        f"Invoice packet for CLM-{cid}. Vendor total: {case['amount']}. "
                        f"Line items reference {case['damage']}."
                    ),
                    metadata={"claim_id": f"CLM-{cid}", "property": case["property"]},
                ),
                Document(
                    doc_id=f"LEASE-{cid}",
                    title=f"Lease excerpt for {case['property']}",
                    doc_type=DocumentType.LEASE,
                    text=(
                        f"Lease for {case['property']} requires tenant cooperation with claim investigations. "
                        f"Alterations and tenant improvements must be scheduled in writing."
                    ),
                    metadata={"claim_id": f"CLM-{cid}", "property": case["property"]},
                ),
            ]
        )
    return docs


def load_eval_questions() -> List[EvalQuestion]:
    questions: List[EvalQuestion] = []
    templates = [
        "Does claim CLM-{id} appear covered?",
        "Should the insurer cover the {damage} claim for {property}?",
        "What is the likely coverage decision for CLM-{id}, and why?",
        "Based on the policy and inspection, is the {cause} loss covered?",
        "Can the adjuster approve CLM-{id} without human review?",
        "Does the evidence support coverage for {property}?",
        "Is there an exclusion that defeats claim CLM-{id}?",
        "What decision should be made for the {property} claim?",
        "Does the inspection support the reported cause for CLM-{id}?",
        "Is the invoice-backed loss at {property} payable under the policy?",
    ]
    for case in CASES:
        label = DecisionLabel(case["policy"])
        for i, template in enumerate(templates, start=1):
            qid = f"Q-{case['id']}-{i:02d}"
            questions.append(
                EvalQuestion(
                    question_id=qid,
                    question=template.format(**case),
                    claim_id=f"CLM-{case['id']}",
                    expected_label=label,
                    required_doc_ids=[f"POL-{case['id']}", f"CLM-{case['id']}", f"INS-{case['id']}"],
                    required_terms=[case["cause"], case["exclusion"].split()[0], case["property"].split()[0]],
                    answer_regex=label.value,
                )
            )
    return questions
