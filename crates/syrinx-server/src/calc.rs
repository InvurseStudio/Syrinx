//! The `calculate` tool: exact arithmetic, so totals don't depend on the model's maths.
//! A port of the Mac's `server/calculator.py`, with its rules and its answers: numbers
//! and arithmetic only (+ - * / // % **, brackets, and round, abs, min, max, sum), worked
//! out in decimal so money adds up to the cent; "$1,234.50", "€12" and "15%" read as
//! people write them. Nothing in it can call code.
//!
//! Small models slip on sums of more than a few numbers ($3,354.11 for $3,054.11), and
//! asking them to check their own work doesn't catch it.

const MAX_EXPRESSION: usize = 2_000;
const MAX_POWER: i128 = 100;
const MAX_EXPRESSIONS: usize = 40;
/// Digits kept after the point in a division that doesn't come out exact (the answer
/// shows at most 12).
const DIVISION_PLACES: u32 = 24;

/// A decimal number: `m` × 10^-`s`. The scale is kept as written ("412.50" has two
/// places), as Python's Decimal keeps it, so money shows its cents.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Dec {
    m: i128,
    s: u32,
}

const TOO_LARGE: &str = "The numbers are too large to work out exactly.";

fn pow10(n: u32) -> Result<i128, String> {
    10i128.checked_pow(n).ok_or_else(|| TOO_LARGE.to_string())
}

impl Dec {
    fn int(n: i128) -> Dec {
        Dec { m: n, s: 0 }
    }

    /// Both at the larger scale.
    fn aligned(a: Dec, b: Dec) -> Result<(i128, i128, u32), String> {
        let s = a.s.max(b.s);
        let am = a.m.checked_mul(pow10(s - a.s)?).ok_or(TOO_LARGE)?;
        let bm = b.m.checked_mul(pow10(s - b.s)?).ok_or(TOO_LARGE)?;
        Ok((am, bm, s))
    }

    fn add(self, b: Dec) -> Result<Dec, String> {
        let (x, y, s) = Dec::aligned(self, b)?;
        Ok(Dec {
            m: x.checked_add(y).ok_or(TOO_LARGE)?,
            s,
        })
    }

    fn neg(self) -> Dec {
        Dec {
            m: -self.m,
            s: self.s,
        }
    }

    fn mul(self, b: Dec) -> Result<Dec, String> {
        let mut d = Dec {
            m: self.m.checked_mul(b.m).ok_or(TOO_LARGE)?,
            s: self.s + b.s,
        };
        // Long products are cut back (rounded) to what can still be shown.
        while d.s > DIVISION_PLACES {
            d = d.rounded(d.s - 1);
        }
        Ok(d)
    }

    /// `self / b`: exact where it comes out exact (100.00 / 4 is 25.00, at the scale the
    /// dividend's places call for), else to `DIVISION_PLACES`, rounded half up.
    fn div(self, b: Dec) -> Result<Dec, String> {
        if b.m == 0 {
            return Err("Division by zero.".into());
        }
        // (m1 / 10^s1) / (m2 / 10^s2) = m1 × 10^(s2 + P) / (m2 × 10^s1), at scale P.
        let num = self
            .m
            .checked_mul(pow10(b.s + DIVISION_PLACES)?)
            .ok_or(TOO_LARGE)?;
        let den = b.m.checked_mul(pow10(self.s)?).ok_or(TOO_LARGE)?;
        let mut q = num / den;
        let r = num % den;
        if r != 0 && 2 * r.unsigned_abs() >= den.unsigned_abs() {
            q += if (num < 0) != (den < 0) { -1 } else { 1 };
        }
        let mut d = Dec {
            m: q,
            s: DIVISION_PLACES,
        };
        // Trailing zeros go, down to the scale an exact answer would have.
        let ideal = self.s.saturating_sub(b.s);
        while d.s > ideal && d.m % 10 == 0 {
            d.m /= 10;
            d.s -= 1;
        }
        Ok(d)
    }

    /// `//`: the quotient cut toward zero, as Decimal's is.
    fn floor_div(self, b: Dec) -> Result<Dec, String> {
        let (x, y, _) = Dec::aligned(self, b)?;
        if y == 0 {
            return Err("Division by zero.".into());
        }
        Ok(Dec::int(x / y))
    }

    /// `%`: the remainder with the dividend's sign, as Decimal's is.
    fn rem(self, b: Dec) -> Result<Dec, String> {
        let (x, y, s) = Dec::aligned(self, b)?;
        if y == 0 {
            return Err("Division by zero.".into());
        }
        Ok(Dec { m: x % y, s })
    }

    fn pow(self, e: Dec) -> Result<Dec, String> {
        let (whole, rest) = (e.m / pow10(e.s)?, e.m % pow10(e.s)?);
        if rest != 0 {
            return Err("Only whole powers are supported.".into());
        }
        if whole.abs() > MAX_POWER {
            return Err(format!("Powers above {MAX_POWER} are not supported."));
        }
        let mut result = Dec::int(1);
        for _ in 0..whole.abs() {
            result = result.mul(self)?;
        }
        if whole < 0 {
            result = Dec::int(1).div(result)?;
        }
        Ok(result)
    }

    /// Rounded half away from zero to `places` places (ROUND_HALF_UP).
    fn rounded(self, places: u32) -> Dec {
        if places >= self.s {
            let m = self.m * 10i128.pow(places - self.s);
            return Dec { m, s: places };
        }
        let cut = 10i128.pow(self.s - places);
        let (q, r) = (self.m / cut, self.m % cut);
        let q = if 2 * r.unsigned_abs() >= cut.unsigned_abs() {
            q + self.m.signum()
        } else {
            q
        };
        Dec { m: q, s: places }
    }

    fn cmp(self, b: Dec) -> std::cmp::Ordering {
        match Dec::aligned(self, b) {
            Ok((x, y, _)) => x.cmp(&y),
            // Too large to align: their signs and sizes decide.
            Err(_) => (self.m as f64 / 10f64.powi(self.s as i32))
                .total_cmp(&(b.m as f64 / 10f64.powi(b.s as i32))),
        }
    }

    /// As the Mac shows it: thousands commas; money keeps its cents ("1,217.05"), "2.5"
    /// stays as it is, long divisions are cut to 12 places and trailing zeros dropped.
    fn shown(self) -> String {
        let digits = self.m.unsigned_abs().to_string();
        let s = self.s as usize;
        let (whole, fraction) = if digits.len() > s {
            (
                digits[..digits.len() - s].to_string(),
                digits[digits.len() - s..].to_string(),
            )
        } else {
            ("0".to_string(), format!("{digits:0>s$}"))
        };
        let mut fraction: String = fraction.chars().take(12).collect();
        if fraction.len() > 2 {
            let trimmed = fraction.trim_end_matches('0');
            fraction = if trimmed.is_empty() {
                String::new()
            } else {
                format!("{trimmed:0<2}")
            };
        }
        if !fraction.is_empty() && fraction.chars().all(|c| c == '0') && fraction.len() != 2 {
            fraction.clear();
        }
        let mut grouped = String::new();
        for (i, c) in whole.chars().enumerate() {
            if i > 0 && (whole.len() - i) % 3 == 0 {
                grouped.push(',');
            }
            grouped.push(c);
        }
        let sign = if self.m < 0 { "-" } else { "" };
        if fraction.is_empty() {
            format!("{sign}{grouped}")
        } else {
            format!("{sign}{grouped}.{fraction}")
        }
    }
}

/// What people write, made into plain arithmetic: × ÷ −, thousands commas ("$1,234.50",
/// "1,234.50"; in a list or a call a bare "100,200" stays two numbers), currency signs,
/// and "15%" as (15/100) where no number follows (then % is the remainder).
fn prepared(expression: &str) -> String {
    let text: Vec<char> = expression
        .trim()
        .replace('×', "*")
        .replace('÷', "/")
        .replace('−', "-")
        .chars()
        .collect();
    let currency = |c: char| matches!(c, '$' | '€' | '£' | '¥');
    // Whether the innermost bracket open before `end` is a list or a call's (…).
    let in_values = |end: usize| {
        let mut opened: Vec<bool> = Vec::new();
        for i in 0..end {
            match text[i] {
                '[' => opened.push(true),
                '(' => {
                    let before = text[..i].iter().rev().find(|c| !c.is_whitespace());
                    opened.push(before.is_some_and(|c| c.is_alphabetic()));
                }
                ')' | ']' => {
                    opened.pop();
                }
                _ => {}
            }
        }
        opened.last().copied().unwrap_or(false)
    };
    let mut out = String::new();
    let mut i = 0;
    while i < text.len() {
        // A grouped number: an optional currency sign, 1-3 digits, then ",ddd" groups,
        // an optional fraction, and no digit after.
        let start = i;
        let sign = currency(text[i]);
        let mut j = if sign { i + 1 } else { i };
        let digits_start = j;
        let at_number_start =
            start == 0 || !(text[start - 1].is_ascii_digit() || text[start - 1] == '.');
        while j < text.len() && text[j].is_ascii_digit() {
            j += 1;
        }
        let lead = j - digits_start;
        let mut groups = 0;
        if at_number_start && (1..=3).contains(&lead) {
            // ",ddd" groups, each with no digit after it (a group may end the text).
            while j + 3 < text.len()
                && text[j] == ','
                && text[j + 1..=j + 3].iter().all(|c| c.is_ascii_digit())
                && !(j + 4 < text.len() && text[j + 4].is_ascii_digit())
            {
                j += 4;
                groups += 1;
            }
        }
        if groups > 0 {
            let mut end = j;
            let fraction =
                end + 1 < text.len() && text[end] == '.' && text[end + 1].is_ascii_digit();
            if fraction {
                end += 1;
                while end < text.len() && text[end].is_ascii_digit() {
                    end += 1;
                }
            }
            let number: String = text[digits_start..end].iter().collect();
            if !sign && !fraction && in_values(start) {
                out.extend(&text[start..end]);
            } else {
                out.push_str(&number.replace(',', ""));
            }
            i = end;
            continue;
        }
        if sign {
            i += 1;
            continue;
        }
        out.push(text[i]);
        i += 1;
    }
    // "15%" with no number or bracket after it is a percentage.
    let chars: Vec<char> = out.chars().collect();
    let mut result = String::new();
    let mut k = 0;
    while k < chars.len() {
        if chars[k].is_ascii_digit()
            || (chars[k] == '.' && chars.get(k + 1).is_some_and(|c| c.is_ascii_digit()))
        {
            let start = k;
            while k < chars.len()
                && (chars[k].is_ascii_digit() || chars[k] == '.' || chars[k] == '_')
            {
                k += 1;
            }
            let number: String = chars[start..k].iter().collect();
            let mut after = k;
            while after < chars.len() && chars[after].is_whitespace() {
                after += 1;
            }
            let percent = after < chars.len() && chars[after] == '%' && {
                let mut next = after + 1;
                while next < chars.len() && chars[next].is_whitespace() {
                    next += 1;
                }
                !(next < chars.len()
                    && (chars[next].is_ascii_digit() || chars[next] == '.' || chars[next] == '('))
            };
            if percent {
                result.push_str(&format!("({number}/100)"));
                k = after + 1;
            } else {
                result.push_str(&number);
            }
            continue;
        }
        result.push(chars[k]);
        k += 1;
    }
    result
}

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Num(Dec),
    Name(String),
    Op(&'static str),
}

fn tokens(source: &str) -> Result<Vec<Token>, String> {
    let chars: Vec<char> = source.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    let unreadable = || format!("Could not read \"{source}\" as arithmetic.");
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit()
            || (c == '.' && chars.get(i + 1).is_some_and(|d| d.is_ascii_digit()))
        {
            let start = i;
            while i < chars.len()
                && (chars[i].is_ascii_digit() || chars[i] == '.' || chars[i] == '_')
            {
                i += 1;
            }
            let written: String = chars[start..i].iter().filter(|c| **c != '_').collect();
            let (whole, fraction) = written.split_once('.').unwrap_or((&written, ""));
            if fraction.contains('.') {
                return Err(unreadable());
            }
            let digits = format!("{whole}{fraction}");
            let m: i128 = digits.parse().map_err(|_| TOO_LARGE.to_string())?;
            out.push(Token::Num(Dec {
                m,
                s: fraction.len() as u32,
            }));
        } else if c.is_alphabetic() {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            out.push(Token::Name(chars[start..i].iter().collect()));
        } else {
            let two: String = chars[i..(i + 2).min(chars.len())].iter().collect();
            let op = match two.as_str() {
                "**" => Some("**"),
                "//" => Some("//"),
                _ => None,
            };
            if let Some(op) = op {
                out.push(Token::Op(op));
                i += 2;
                continue;
            }
            let op = match c {
                '+' => "+",
                '-' => "-",
                '*' => "*",
                '/' => "/",
                '%' => "%",
                '(' => "(",
                ')' => ")",
                '[' => "[",
                ']' => "]",
                ',' => ",",
                _ => return Err(unreadable()),
            };
            out.push(Token::Op(op));
            i += 1;
        }
    }
    Ok(out)
}

#[derive(Clone, Debug)]
enum Value {
    Num(Dec),
    List(Vec<Dec>),
}

const ALLOWED: &str =
    "Only numbers, + - * / // % **, brackets and round, abs, min, max, sum are allowed.";

struct Parser {
    tokens: Vec<Token>,
    at: usize,
    source: String,
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.at)
    }

    fn eat(&mut self, op: &str) -> bool {
        if self.peek()
            == Some(&Token::Op(match op {
                "+" => "+",
                "-" => "-",
                "*" => "*",
                "/" => "/",
                "//" => "//",
                "%" => "%",
                "**" => "**",
                "(" => "(",
                ")" => ")",
                "[" => "[",
                "]" => "]",
                _ => ",",
            }))
        {
            self.at += 1;
            true
        } else {
            false
        }
    }

    fn unreadable(&self) -> String {
        format!("Could not read \"{}\" as arithmetic.", self.source)
    }

    fn number(v: Value) -> Result<Dec, String> {
        match v {
            Value::Num(d) => Ok(d),
            Value::List(_) => Err("Lists can only be given to sum, min or max.".into()),
        }
    }

    fn expr(&mut self) -> Result<Value, String> {
        let mut left = self.term()?;
        loop {
            if self.eat("+") {
                let right = Parser::number(self.term()?)?;
                left = Value::Num(Parser::number(left)?.add(right)?);
            } else if self.eat("-") {
                let right = Parser::number(self.term()?)?;
                left = Value::Num(Parser::number(left)?.add(right.neg())?);
            } else {
                return Ok(left);
            }
        }
    }

    fn term(&mut self) -> Result<Value, String> {
        let mut left = self.unary()?;
        loop {
            let op = if self.eat("*") {
                "*"
            } else if self.eat("//") {
                "//"
            } else if self.eat("/") {
                "/"
            } else if self.eat("%") {
                "%"
            } else {
                return Ok(left);
            };
            let (a, b) = (Parser::number(left)?, Parser::number(self.unary()?)?);
            left = Value::Num(match op {
                "*" => a.mul(b)?,
                "/" => a.div(b)?,
                "//" => a.floor_div(b)?,
                _ => a.rem(b)?,
            });
        }
    }

    fn unary(&mut self) -> Result<Value, String> {
        if self.eat("-") {
            return Ok(Value::Num(Parser::number(self.unary()?)?.neg()));
        }
        if self.eat("+") {
            return Ok(Value::Num(Parser::number(self.unary()?)?));
        }
        self.power()
    }

    /// `**` binds tighter than a minus on its left (-2**2 is -4) and takes one on its right.
    fn power(&mut self) -> Result<Value, String> {
        let base = self.atom()?;
        if self.eat("**") {
            let exponent = Parser::number(self.unary()?)?;
            return Ok(Value::Num(Parser::number(base)?.pow(exponent)?));
        }
        Ok(base)
    }

    fn values(&mut self, close: &str) -> Result<Vec<Value>, String> {
        let mut items = Vec::new();
        if self.eat(close) {
            return Ok(items);
        }
        loop {
            items.push(self.expr()?);
            if self.eat(close) {
                return Ok(items);
            }
            if !self.eat(",") {
                return Err(self.unreadable());
            }
            // A trailing comma before the close.
            if self.eat(close) {
                return Ok(items);
            }
        }
    }

    fn atom(&mut self) -> Result<Value, String> {
        match self.peek().cloned() {
            Some(Token::Num(d)) => {
                self.at += 1;
                Ok(Value::Num(d))
            }
            Some(Token::Op("(")) => {
                self.at += 1;
                let items = self.values(")")?;
                match items.len() {
                    1 => Ok(items.into_iter().next().unwrap()),
                    0 => Err(self.unreadable()),
                    _ => Ok(Value::List(
                        items
                            .into_iter()
                            .map(Parser::number)
                            .collect::<Result<_, _>>()?,
                    )),
                }
            }
            Some(Token::Op("[")) => {
                self.at += 1;
                let items = self.values("]")?;
                Ok(Value::List(
                    items
                        .into_iter()
                        .map(Parser::number)
                        .collect::<Result<_, _>>()?,
                ))
            }
            Some(Token::Name(name)) => {
                self.at += 1;
                if !matches!(name.as_str(), "round" | "abs" | "min" | "max" | "sum")
                    || !self.eat("(")
                {
                    return Err(ALLOWED.into());
                }
                let args = self.values(")")?;
                call(&name, args)
            }
            _ => Err(self.unreadable()),
        }
    }
}

fn call(name: &str, args: Vec<Value>) -> Result<Value, String> {
    match name {
        "min" | "max" | "sum" => {
            let items: Vec<Dec> = match args.as_slice() {
                [Value::List(items)] => items.clone(),
                _ => args
                    .into_iter()
                    .map(|a| match a {
                        Value::Num(d) => Ok(d),
                        Value::List(_) => Err(format!("{name} needs numbers.")),
                    })
                    .collect::<Result<_, _>>()?,
            };
            if items.is_empty() {
                return Err(format!("{name} needs numbers."));
            }
            let mut acc = if name == "sum" { Dec::int(0) } else { items[0] };
            for (i, item) in items.iter().enumerate() {
                acc = match name {
                    "sum" => acc.add(*item)?,
                    "min" if i > 0 && item.cmp(acc).is_lt() => *item,
                    "max" if i > 0 && item.cmp(acc).is_gt() => *item,
                    _ => acc,
                };
            }
            Ok(Value::Num(acc))
        }
        "abs" if args.len() == 1 => {
            let d = Parser::number(args.into_iter().next().unwrap())?;
            Ok(Value::Num(Dec {
                m: d.m.abs(),
                s: d.s,
            }))
        }
        "round" if (1..=2).contains(&args.len()) => {
            let mut args = args.into_iter();
            let value = Parser::number(args.next().unwrap())?;
            let places = match args.next() {
                Some(p) => {
                    let p = Parser::number(p)?;
                    let whole = p.m / pow10(p.s)?;
                    if !(0..=12).contains(&whole) {
                        return Err("round takes 0 to 12 places.".into());
                    }
                    whole as u32
                }
                None => 0,
            };
            Ok(Value::Num(value.rounded(places)))
        }
        _ => Err(format!("{name} was given the wrong number of values.")),
    }
}

/// "412.50 + 129.30 = 541.80", or what's wrong with the expression.
pub fn calculate(expression: &str) -> Result<String, String> {
    if expression.trim().is_empty() {
        return Err("Give an expression, e.g. 412.50 + 129.30 + 400.".into());
    }
    if expression.chars().count() > MAX_EXPRESSION {
        return Err(format!(
            "The expression is too long (at most {MAX_EXPRESSION} characters)."
        ));
    }
    let source = prepared(expression);
    let mut parser = Parser {
        tokens: tokens(&source)?,
        at: 0,
        source: expression.trim().to_string(),
    };
    let value = parser.expr()?;
    if parser.at != parser.tokens.len() {
        return Err(parser.unreadable());
    }
    match value {
        Value::Num(d) => Ok(format!("{} = {}", expression.trim(), d.shown())),
        Value::List(_) => Err("The result is a list; use sum, min or max on it.".into()),
    }
}

/// Several expressions, one a line (or split by ";"), one result a line; an error on one
/// line doesn't stop the others. One expression gives its result, or its error.
pub fn calculate_many(text: &str) -> Result<String, String> {
    let expressions: Vec<&str> = text
        .split(['\n', ';'])
        .map(str::trim)
        .filter(|e| !e.is_empty())
        .collect();
    match expressions.len() {
        0 => Err("Give at least one expression.".into()),
        1 => calculate(expressions[0]),
        n if n > MAX_EXPRESSIONS => Err(format!("At most {MAX_EXPRESSIONS} expressions at once.")),
        _ => Ok(expressions
            .iter()
            .map(|e| calculate(e).unwrap_or_else(|why| format!("{e} → error: {why}")))
            .collect::<Vec<_>>()
            .join("\n")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is(expression: &str, want: &str) {
        assert_eq!(
            calculate(expression),
            Ok(format!("{expression} = {want}")),
            "{expression}"
        );
    }

    #[test]
    fn money_adds_up_to_the_cent_as_on_the_mac() {
        is("412.50 + 129.30 + 400", "941.80");
        is("1500.00 + 1554.11", "3,054.11");
        is("sum([12.40, 8.99, 31.00]) * 1.13", "59.2007");
        is("$1,234.50 * 2", "2,469.00");
        is("86.40 * 18%", "15.552");
        is("round(86.40 * 18%, 2)", "15.55");
        is("round(2.5)", "3");
        is("round(-2.5)", "-3");
        is("100.00 / 4", "25.00");
        is("10 / 4", "2.5");
        is("1 / 3", "0.333333333333");
        // Cut, not rounded, as the Mac's _shown cuts its 12 places.
        is("2 / 3", "0.666666666666");
        is("10 % 3", "1");
        is("-7 % 3", "-1");
        is("7 // 2", "3");
        is("2 ** 10", "1,024");
        is("-2 ** 2", "-4");
        is("2 ** -1", "0.5");
        is("max(100,200)", "200");
        is("min([3, 1.5, 2])", "1.5");
        is("abs(-12.30)", "12.30");
        is("€12 × 3 − 1", "35");
        is("(2 + 3) * 4", "20");
        is("1,234 + 1", "1,235");
        is("0.1 + 0.2", "0.3");
        is("1000000 * 1000000", "1,000,000,000,000");
    }

    #[test]
    fn nothing_but_arithmetic_is_read() {
        assert_eq!(calculate("1 / 0"), Err("Division by zero.".into()));
        assert!(
            calculate("exec(1)")
                .unwrap_err()
                .starts_with("Only numbers")
        );
        assert!(calculate("__import__('os')").is_err());
        assert!(calculate("2 +").unwrap_err().starts_with("Could not read"));
        assert!(calculate("[1, 2]").unwrap_err().contains("list"));
        assert!(calculate("2 ** 1000").unwrap_err().contains("Powers above"));
        assert!(calculate("").is_err());
    }

    #[test]
    fn several_at_once_one_a_line() {
        assert_eq!(
            calculate_many("1 + 1\n2 * 3; 1 / 0"),
            Ok("1 + 1 = 2\n2 * 3 = 6\n1 / 0 → error: Division by zero.".into())
        );
        assert_eq!(calculate_many("5 * 5"), Ok("5 * 5 = 25".into()));
        assert!(calculate_many("1 / 0").is_err());
    }
}
