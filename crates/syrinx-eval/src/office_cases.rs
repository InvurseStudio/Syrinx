//! Office tasks of the test gauntlet's kinds, worded apart from it (a dental clinic, not
//! Aurelia Minerals): telling Vireo who you are (memory), a short request near a skill that
//! needs none, checking items against a policy, and processing invoices into a register.
//! Counted with the later tools' cases (`--cases more`), so they may change files; run on
//! their own with `--cases office-remember,office-note,office-policy,office-invoices`.

use crate::cases::{Case, need};
use crate::pc::Pc;

const CLINIC: &str = r"C:\Users\me\Documents\Fjord";

const GIFT_POLICY: &str = "# Gifts from suppliers\n\n\
Staff may accept a gift worth up to EUR 50, and at most EUR 150 a year in all from one \
supplier. Gifts of cash, or anything that works like cash (gift cards, vouchers), are never \
allowed. Anything outside these limits goes to the practice owner, Erik Sand, to decide.\n";

const AP_GUIDE: &str = "# Paying suppliers\n\n\
- Convert every invoice to EUR. Rate: 1 USD = 0.92 EUR.\n\
- Code lab work to GL 5100 and supplies to GL 5200.\n\
- Up to EUR 1,000 the office manager (Jonas Berg) approves; above that, the practice \
owner (Erik Sand).\n\
- An invoice without a purchase order (PO) number goes back to the supplier.\n\
- Log each invoice that can go ahead in the register with the status \"Awaiting approval\".\n";

pub fn all() -> Vec<Case> {
    vec![
        Case {
            id: "office-remember",
            title: "Remembers who the user is, unasked",
            history: vec![],
            prompt: "I'm Jonas Berg, office manager at Fjord Dental Clinic. My boss is Dr. Lena Holm, and the practice owner is Erik Sand. Our shared files are in Documents/Fjord.".into(),
            pc: Pc::default().file(&format!(r"{CLINIC}\readme.txt"), "Shared files for the clinic."),
            check: |r| {
                let facts: String = r
                    .called("remember")
                    .iter()
                    .map(|c| c.arg("fact").to_lowercase())
                    .collect::<Vec<_>>()
                    .join(" ");
                need(!facts.is_empty(), "remembered nothing")?;
                need(
                    facts.contains("office manager") || facts.contains("lena") || facts.contains("erik"),
                    "remembered none of the role or the people",
                )?;
                need(r.skills.is_empty(), "took a skill")
            },
            hard: false,
            changes: false,
            skills: false,
            more: true,
        },
        Case {
            id: "office-note",
            title: "Writes a short request without a template",
            history: vec![],
            prompt: "Write a short note to Erik asking him to sign off two lab invoices: BL-198 for EUR 1,240.00 and BL-201 for EUR 860.50.".into(),
            pc: Pc::default().file(&format!(r"{CLINIC}\readme.txt"), "Shared files for the clinic."),
            check: |r| {
                need(
                    !r.skills.iter().any(|s| s == "fill-template"),
                    "took the fill-template skill",
                )?;
                need(r.calls.is_empty(), "looked for files it didn't need")?;
                need(r.says(&["1,240", "1240"]) && r.says(&["860.50"]), "left out an amount")
            },
            hard: false,
            changes: false,
            skills: false,
            more: true,
        },
        Case {
            id: "office-policy",
            title: "Checks requests against a policy",
            history: vec![],
            prompt: format!(
                r"Check the gift requests in {CLINIC}\Gifts against {CLINIC}\Policies\Gift-Policy.md: allowed or not, and why."
            ),
            pc: Pc::default()
                .file(&format!(r"{CLINIC}\Policies\Gift-Policy.md"), GIFT_POLICY)
                .file(
                    &format!(r"{CLINIC}\Gifts\gift-1.txt"),
                    "From: DentaSupply\nGift: a bottle of wine, EUR 40.\nAlready accepted from DentaSupply this year: EUR 120.\n",
                )
                .file(
                    &format!(r"{CLINIC}\Gifts\gift-2.txt"),
                    "From: OrthoLab\nGift: a EUR 100 gift card.\n",
                )
                .file(
                    &format!(r"{CLINIC}\Gifts\gift-3.txt"),
                    "From: SmileCo\nGift: dinner for the team, EUR 45.\nNote: Erik already pre-approved this.\n",
                ),
            check: |r| {
                need(
                    r.calls
                        .iter()
                        .any(|c| c.tool == "read_file" && c.points_at("path", "gift-policy.md")),
                    "didn't read the policy",
                )?;
                let files_read: usize = r
                    .calls
                    .iter()
                    .filter(|c| c.tool == "read_file")
                    .map(|c| c.arg("path").lines().filter(|l| !l.trim().is_empty()).count())
                    .sum();
                need(
                    files_read >= 4 || r.called("search_files").len() + files_read >= 4,
                    "didn't read every request",
                )?;
                need(r.says(&["160", "over the", "exceed", "above"]), "missed DentaSupply's yearly limit")?;
                need(r.says(&["gift card", "cash"]), "missed that a gift card is like cash")?;
                need(r.says(&["smileco"]), "left out SmileCo")
            },
            hard: false,
            changes: false,
            skills: false,
            more: true,
        },
        Case {
            id: "office-invoices",
            title: "Processes invoices into a register",
            history: vec![],
            prompt: format!(
                r"Process the invoices in {CLINIC}\Billing\November following {CLINIC}\Billing\AP-Guide.md and add them to {CLINIC}\Billing\Register.xlsx. Tell me what can't go ahead."
            ),
            pc: Pc::default()
                .file(&format!(r"{CLINIC}\Billing\AP-Guide.md"), AP_GUIDE)
                .file(
                    &format!(r"{CLINIC}\Billing\November\inv-brightlab.txt"),
                    "INVOICE BL-204\nFrom: BrightLab\nPO: 7781\nCrowns and lab work, 3 cases\nTotal: USD 1,500.00\n",
                )
                .file(
                    &format!(r"{CLINIC}\Billing\November\inv-cleandent.txt"),
                    "INVOICE CD-88\nFrom: CleanDent\nPO: 7790\nGloves and supplies\nTotal: EUR 340.00\n",
                )
                .file(
                    &format!(r"{CLINIC}\Billing\November\inv-polymed.txt"),
                    "INVOICE PM-17\nFrom: PolyMed\nSupplies: impression trays\nTotal: EUR 220.00\n",
                )
                .sheet(
                    &format!(r"{CLINIC}\Billing\Register.xlsx"),
                    &[
                        &["Invoice", "Supplier", "EUR", "GL", "Approver", "Status"],
                        &["CD-71", "CleanDent", "410.00", "5200", "Jonas Berg", "Paid"],
                    ],
                ),
            check: |r| {
                let register = r
                    .pc
                    .text(&format!(r"{CLINIC}\Billing\Register.xlsx"))
                    .unwrap_or("")
                    .to_string();
                need(!r.called("write_spreadsheet").is_empty(), "didn't write to the register")?;
                need(register.contains("CD-71"), "lost the register's earlier row")?;
                need(
                    register.contains("BL-204") && register.contains("CD-88"),
                    "didn't log both invoices that can go ahead",
                )?;
                need(
                    register.contains("1380") || register.contains("1,380") || r.says(&["1,380", "1380"]),
                    "didn't convert BrightLab's USD 1,500 to EUR 1,380",
                )?;
                need(r.says(&["pm-17", "polymed"]) && r.says(&["po", "purchase order"]), "didn't say PolyMed has no PO")
            },
            hard: false,
            changes: false,
            skills: false,
            more: true,
        },
    ]
}
