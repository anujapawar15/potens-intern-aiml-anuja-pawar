"""
Generates 6 synthetic multi-page PDF policy documents for a fictional
company ("Nimbus Retail Technologies") so the RAG app is runnable end-to-end
without requiring the user to source their own documents.

Two of the documents (remote_work_policy_2023.pdf and
remote_work_policy_2024.pdf) intentionally state conflicting remote-work
rules -- this gives the /contradict endpoint something real to detect.

Run: python scripts/generate_sample_pdfs.py
"""
import sys
from pathlib import Path

from fpdf import FPDF

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "data" / "pdfs"


def make_pdf(filename: str, title: str, pages: list[str]) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for page_text in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.multi_cell(0, 10, title)
        pdf.ln(2)
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 7, page_text)
    out_path = OUT_DIR / filename
    pdf.output(str(out_path))
    print(f"  wrote {out_path}")


DOCUMENTS = [
    (
        "remote_work_policy_2023.pdf",
        "Nimbus Retail Technologies - Remote Work Policy (Effective Jan 2023)",
        [
            (
                "1. Purpose\n"
                "This policy defines the rules under which Nimbus Retail Technologies "
                "employees may work remotely instead of attending the office in person.\n\n"
                "2. Eligibility\n"
                "All full-time employees who have completed their 90-day probation period "
                "are eligible to request remote work arrangements. Contractors and interns "
                "are not covered by this policy and must work on-site unless separately "
                "approved by HR.\n\n"
                "3. Remote Work Allowance\n"
                "Eligible employees may work remotely for a maximum of three (3) days per "
                "week. The remaining two (2) days must be spent working from a company "
                "office. Managers may approve temporary exceptions for medical or family "
                "emergencies, capped at four (4) consecutive weeks."
            ),
            (
                "4. Equipment\n"
                "Employees working remotely are provided with a company laptop, a docking "
                "station, and a one-time home-office stipend of INR 10,000 to cover desk "
                "and chair costs. Requests for additional equipment must go through the "
                "IT helpdesk.\n\n"
                "5. Core Hours\n"
                "Regardless of location, employees must be reachable on Slack and email "
                "between 10:00 AM and 4:00 PM IST, which are considered core collaboration "
                "hours.\n\n"
                "6. Review\n"
                "This policy will be reviewed annually by the People Operations team and "
                "may be revised based on business needs and employee feedback."
            ),
        ],
    ),
    (
        "remote_work_policy_2024.pdf",
        "Nimbus Retail Technologies - Remote Work Policy (Effective Feb 2024, Revised)",
        [
            (
                "1. Purpose and Revision Notice\n"
                "This document supersedes the Remote Work Policy dated January 2023. "
                "Following a company-wide review of collaboration metrics, Nimbus Retail "
                "Technologies is moving to an office-first model.\n\n"
                "2. Eligibility\n"
                "All full-time employees who have completed their 90-day probation period "
                "remain eligible to request occasional remote work, subject to the new "
                "limits below. Contractors and interns must continue to work on-site.\n\n"
                "3. Remote Work Allowance (Revised)\n"
                "Effective February 2024, employees are required to work from a company "
                "office at least four (4) days per week. Remote work is limited to a "
                "maximum of one (1) day per week and must be pre-approved by the "
                "employee's direct manager at least 24 hours in advance."
            ),
            (
                "4. Rationale\n"
                "Internal surveys showed a measurable decline in cross-team collaboration "
                "and onboarding speed for new hires under the previous three-day remote "
                "allowance. Leadership has therefore reduced the remote allowance to one "
                "day per week starting this quarter.\n\n"
                "5. Equipment\n"
                "The home-office stipend and company laptop program remain unchanged from "
                "the 2023 policy.\n\n"
                "6. Review\n"
                "This revised policy will be reassessed after two quarters of data "
                "collection, no later than September 2024."
            ),
        ],
    ),
    (
        "leave_policy.pdf",
        "Nimbus Retail Technologies - Leave and Holiday Policy",
        [
            (
                "1. Annual Leave\n"
                "Every full-time employee accrues eighteen (18) days of paid annual leave "
                "per calendar year, credited at 1.5 days per completed month of service. "
                "Unused annual leave may be carried forward up to a maximum of ten (10) "
                "days into the following year; any balance beyond that is forfeited.\n\n"
                "2. Sick Leave\n"
                "Employees are entitled to twelve (12) days of paid sick leave per year. "
                "A medical certificate is required for sick leave exceeding three (3) "
                "consecutive days.\n\n"
                "3. Maternity and Paternity Leave\n"
                "Nimbus Retail Technologies offers twenty-six (26) weeks of paid maternity "
                "leave and two (2) weeks of paid paternity leave, in line with applicable "
                "labor regulations."
            ),
            (
                "4. Public Holidays\n"
                "Employees receive twelve (12) paid public holidays per year based on the "
                "regional holiday calendar published by HR every January.\n\n"
                "5. Leave Application Process\n"
                "All leave requests must be submitted through the HR portal at least three "
                "(3) working days in advance, except for sick leave, which may be reported "
                "on the day of absence. Approval rests with the employee's direct manager.\n\n"
                "6. Unpaid Leave\n"
                "Employees who exhaust their paid leave balance may request unpaid leave "
                "of up to thirty (30) days per year, subject to manager and HR approval."
            ),
        ],
    ),
    (
        "code_of_conduct.pdf",
        "Nimbus Retail Technologies - Code of Conduct",
        [
            (
                "1. Purpose\n"
                "This Code of Conduct sets the standard of behavior expected from every "
                "employee, contractor, and vendor representing Nimbus Retail Technologies.\n\n"
                "2. Respectful Workplace\n"
                "Nimbus Retail Technologies is committed to a workplace free of "
                "harassment, discrimination, and retaliation. Any employee who experiences "
                "or witnesses such behavior should report it to HR or through the "
                "anonymous ethics hotline within thirty (30) days of the incident.\n\n"
                "3. Conflicts of Interest\n"
                "Employees must disclose any outside employment, financial interest, or "
                "personal relationship that could reasonably be seen to conflict with "
                "their duties at Nimbus Retail Technologies."
            ),
            (
                "4. Confidentiality\n"
                "Employees must not disclose confidential company information, including "
                "customer data, financial results, or unreleased product plans, to anyone "
                "outside the company without prior written authorization.\n\n"
                "5. Gifts and Entertainment\n"
                "Employees may not accept gifts from vendors or partners with a value "
                "exceeding INR 2,000 without disclosing the gift to their manager.\n\n"
                "6. Disciplinary Action\n"
                "Violations of this Code of Conduct may result in disciplinary action up "
                "to and including termination of employment, depending on the severity "
                "and frequency of the violation."
            ),
        ],
    ),
    (
        "it_security_policy.pdf",
        "Nimbus Retail Technologies - IT Security Policy",
        [
            (
                "1. Password Requirements\n"
                "All company accounts must use passwords that are at least twelve (12) "
                "characters long and include uppercase letters, lowercase letters, "
                "numbers, and symbols. Passwords must be changed every ninety (90) days "
                "and must not be reused across the last five (5) passwords.\n\n"
                "2. Multi-Factor Authentication\n"
                "Multi-factor authentication (MFA) is mandatory for all employees "
                "accessing company email, VPN, and internal systems from outside the "
                "office network.\n\n"
                "3. Device Security\n"
                "Company laptops must have disk encryption and endpoint protection "
                "software enabled at all times. Lost or stolen devices must be reported "
                "to IT Security within one (1) hour of discovery."
            ),
            (
                "4. Data Classification\n"
                "Company data is classified as Public, Internal, Confidential, or "
                "Restricted. Confidential and Restricted data must not be stored on "
                "personal devices or personal cloud storage accounts.\n\n"
                "5. Acceptable Use\n"
                "Company systems are provided for business use. Limited personal use is "
                "permitted as long as it does not interfere with work duties or violate "
                "other company policies.\n\n"
                "6. Incident Reporting\n"
                "Suspected security incidents, including phishing emails and suspicious "
                "login attempts, must be reported immediately to security@nimbusretail.com."
            ),
        ],
    ),
    (
        "expense_reimbursement_policy.pdf",
        "Nimbus Retail Technologies - Travel and Expense Reimbursement Policy",
        [
            (
                "1. Eligible Expenses\n"
                "Employees traveling for approved business purposes may claim "
                "reimbursement for airfare, ground transportation, lodging, and meals, "
                "subject to the daily limits described in Section 3.\n\n"
                "2. Pre-Approval\n"
                "All business travel exceeding INR 20,000 in total estimated cost "
                "requires written pre-approval from the employee's manager and the "
                "finance team before booking.\n\n"
                "3. Daily Limits\n"
                "Meal expenses are capped at INR 1,500 per day for domestic travel and "
                "INR 3,000 per day for international travel. Lodging is capped at INR "
                "8,000 per night domestically, based on actual receipts."
            ),
            (
                "4. Submission Process\n"
                "Expense claims must be submitted through the finance portal within "
                "fourteen (14) days of the trip's completion, along with itemized "
                "receipts for any single expense over INR 500.\n\n"
                "5. Non-Reimbursable Items\n"
                "Alcohol, entertainment unrelated to business purposes, and traffic "
                "fines are not reimbursable under any circumstances.\n\n"
                "6. Reimbursement Timeline\n"
                "Approved expense claims are reimbursed with the employee's next payroll "
                "cycle, typically within fifteen (15) business days of approval."
            ),
        ],
    ),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating {len(DOCUMENTS)} sample PDFs into {OUT_DIR} ...")
    for filename, title, pages in DOCUMENTS:
        make_pdf(filename, title, pages)
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
