"""Synthetic support tickets: the dataset the factory trains and evaluates on.

Eight categories, generated from templates with slot filling and a seeded RNG, so every
container and every attendee gets exactly the same split without downloading anything.
Small on purpose: the eval set has to run in seconds on a T4, and the train set has to
fine-tune in minutes.

    from tickets import load_split
    train = load_split("train")   # 800 tickets
    test = load_split("test")     # 120 tickets
"""

from __future__ import annotations

import random
from dataclasses import dataclass

CATEGORIES = [
    "billing",
    "refund",
    "shipping",
    "account_access",
    "bug_report",
    "feature_request",
    "cancellation",
    "other",
]

# Dataset versions. "v1" is the eight categories above. "v2" is what happens on day two:
# the support team adds a category the model in production has never seen.
DATASET_VERSIONS: dict[str, list[str]] = {
    "v1": CATEGORIES,
    "v2": [*CATEGORIES, "data_request"],
}

# Slot values shared across templates.
PRODUCTS = [
    "the Pro plan",
    "the Team plan",
    "my subscription",
    "the mobile app",
    "the desktop app",
    "the API",
    "the dashboard",
    "the Starter plan",
]
ITEMS = [
    "the standing desk",
    "the desk lamp",
    "the headphones",
    "the replacement cable",
    "the monitor arm",
    "the keyboard",
    "the chair",
    "the two lamps",
]
ORDERS = [f"#{n}" for n in range(48210, 48310)]
AMOUNTS = ["$12", "$29", "$49", "$120", "$240", "$9.99", "$35.50", "$300"]
DAYS = ["3 days", "a week", "10 days", "two weeks", "yesterday", "last Monday", "a month"]
BROWSERS = ["Chrome", "Safari", "Firefox", "Edge", "the iOS app", "the Android app"]
FEATURES = [
    "dark mode",
    "CSV export",
    "two-factor auth",
    "a Slack integration",
    "bulk editing",
    "saved filters",
    "an audit log",
    "keyboard shortcuts",
]
NAMES = ["Sam", "Priya", "Jordan", "Alex", "Mei", "Tomasz", "Lena", "Diego"]

TEMPLATES: dict[str, list[str]] = {
    "billing": [
        "I was charged {amount} twice this month for {product}. Can you check my invoices?",
        "Why did my bill go up to {amount}? I didn't change anything on {product}.",
        "The invoice for {product} shows {amount} but the pricing page says less. Please explain.",
        "Can I get a receipt for the {amount} payment from {days} ago?",
        "My card was declined but you still charged me {amount}. What's going on with my billing?",
        "Do you support annual billing for {product}? Would like to switch and pay {amount} up front.",
        "The VAT on my last invoice looks wrong, I paid {amount}. Can someone review it?",
    ],
    "refund": [
        "I'd like a refund for {item}, order {order}. It arrived damaged.",
        "Please refund the {amount} I paid {days} ago for {product}. I never used it.",
        "Order {order} was cancelled but I still haven't received my {amount} back.",
        "Requesting a refund for {item}: it's not what was described on the site.",
        "I returned {item} ({order}) {days} ago. When will the refund show up?",
        "Can I get my money back for {product}? It doesn't do what I need.",
    ],
    "shipping": [
        "Where is my order {order}? It was supposed to arrive {days} ago.",
        "Tracking for {item} hasn't updated in {days}. Is it lost?",
        "Can you change the delivery address for order {order}? I moved.",
        "Order {order} says delivered but I never got {item}.",
        "How long does shipping take for {item} to Canada?",
        "I need {item} by Friday. Can order {order} be expedited?",
    ],
    "account_access": [
        "I can't log in. The password reset email for my account never arrives.",
        "My account got locked after too many attempts. Please unlock it, it's {name}@example.com.",
        "Two-factor codes aren't working since I got a new phone. How do I get back in?",
        "I get 'invalid credentials' on {browser} even though the password is right.",
        "Someone changed the email on my account and I'm locked out. Please help.",
        "How do I recover my account? I lost access to the email I signed up with.",
    ],
    "bug_report": [
        "The export button on the dashboard does nothing in {browser}. Console shows a 500 error.",
        "{product} crashes every time I open the settings page since the update {days} ago.",
        "Dates in the reports are off by one day after the {browser} update.",
        "Search returns no results for anything with an apostrophe. Reproducible on {browser}.",
        "Uploading a file larger than 10MB fails silently in {product}.",
        "The API returns 502 intermittently on /v1/reports since {days} ago.",
        "Notifications show the wrong timezone in {product}. I'm in UTC+2.",
    ],
    "feature_request": [
        "It would be great if {product} had {feature}.",
        "Any plans to add {feature}? It's the one thing keeping us from upgrading.",
        "Please consider {feature} for {product}. Our whole team is asking for it.",
        "Feature idea: {feature}. Happy to beta test it.",
        "Is {feature} on the roadmap? We'd pay more for {product} if it had it.",
    ],
    "cancellation": [
        "Please cancel {product}. I don't need it anymore.",
        "How do I cancel my subscription before the next {amount} charge?",
        "I want to close my account and cancel {product} effective immediately.",
        "Cancel order {order} please, I ordered {item} by mistake.",
        "We're moving to another vendor. Please cancel {product} at the end of the term.",
    ],
    "other": [
        "Do you have a partner program? We're an agency and would like to resell {product}.",
        "Hi {name} here, just wanted to say the support last week was excellent. Thanks!",
        "Is there a student discount for {product}?",
        "Where can I find the terms of service and your data processing agreement?",
        "Can I speak to someone about a press inquiry regarding {product}?",
        "What's the difference between {product} and the Enterprise plan?",
        "Are you hiring? I'd love to work on {product}.",
    ],
    # v2 only: privacy and data-subject requests, which used to be misfiled under "other".
    "data_request": [
        "Under GDPR I'd like a copy of all the personal data you hold on me ({name}@example.com).",
        "Please delete my account and all associated data. This is a formal erasure request.",
        "Can you export everything you store about my account? I'm moving to a competitor.",
        "I want to know which third parties you've shared my data with in the last year.",
        "Data subject access request: please send me my records within 30 days.",
        "Please stop processing my data for marketing and confirm in writing.",
        "How do I download all my {product} data before I close my account?",
    ],
}

# Paraphrases that only appear in the held-out split, so a model cannot ace the eval by
# memorizing the training templates.
TEST_ONLY_TEMPLATES: dict[str, list[str]] = {
    "billing": [
        "There's a {amount} line on my statement I don't recognize. Is that from {product}?",
        "Your billing page and my card statement disagree by {amount}. Which one is right?",
    ],
    "refund": [
        "I was promised my {amount} back {days} ago and nothing has landed. Please sort out the refund.",
        "{item} from order {order} is faulty. I want my money back, not a replacement.",
    ],
    "shipping": [
        "The courier says order {order} is out for delivery but that was {days} ago. Where is it?",
        "Can {item} ship to a PO box? Order {order} got rejected at checkout for the address.",
    ],
    "account_access": [
        "Password reset link says expired the second I click it. Can't get into my account at all.",
        "Login keeps bouncing me back to the sign-in page on {browser} with no error.",
    ],
    "bug_report": [
        "Since {days} ago the {browser} version throws 'undefined is not a function' when I save.",
        "Report totals don't add up: the summary row is off by exactly one row's worth in {product}.",
    ],
    "feature_request": [
        "Would love {feature} in {product}. Right now we hack around it with spreadsheets.",
        "Adding {feature} would make {product} a no-brainer for our team.",
    ],
    "cancellation": [
        "Please stop my {product} renewal. I won't be needing it after this month.",
        "Terminate my subscription to {product}; don't charge me the next {amount}.",
    ],
    "other": [
        "Who do I talk to about a bulk purchase of {product} for a school district?",
        "Do you publish a changelog for {product}? Couldn't find one on the site.",
    ],
    "data_request": [
        "CCPA request: tell me what personal information of mine you sell or share, and opt me out.",
        "I need my full data export for a legal matter. Who handles privacy requests?",
    ],
}


def system_prompt(version: str = "v1") -> str:
    return (
        "You are a support ticket router. Classify the ticket into exactly one category from this list: "
        + ", ".join(DATASET_VERSIONS[version])
        + ". Reply with the category name only."
    )


SYSTEM_PROMPT = system_prompt("v1")


@dataclass(frozen=True)
class Ticket:
    id: str
    text: str
    label: str


def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        product=rng.choice(PRODUCTS),
        item=rng.choice(ITEMS),
        order=rng.choice(ORDERS),
        amount=rng.choice(AMOUNTS),
        days=rng.choice(DAYS),
        browser=rng.choice(BROWSERS),
        feature=rng.choice(FEATURES),
        name=rng.choice(NAMES),
    )


def generate(
    n: int, seed: int, extra: dict[str, list[str]] | None = None, categories: list[str] = CATEGORIES
) -> list[Ticket]:
    """`n` tickets, balanced across categories, deterministic for a seed."""
    rng = random.Random(seed)
    tickets = []
    per_cat = n // len(categories)
    for cat in categories:
        pool = TEMPLATES[cat] + (extra or {}).get(cat, [])
        for i in range(per_cat):
            text = _fill(rng.choice(pool), rng)
            # Light noise so the model cannot memorize templates verbatim.
            if rng.random() < 0.3:
                text = rng.choice(["Hi, ", "Hello, ", "Hey team, ", "Urgent: ", ""]) + text
            if rng.random() < 0.3:
                text = text + rng.choice(
                    [" Thanks.", " Please advise.", " Thank you!", " Regards, " + rng.choice(NAMES), ""]
                )
            tickets.append(Ticket(id=f"T-{seed}-{cat[:3]}-{i:03d}", text=text, label=cat))
    rng.shuffle(tickets)
    return tickets


def load_split(split: str, version: str = "v1") -> list[Ticket]:
    """800 training / 126 held-out tickets for a dataset version (v2 has nine categories)."""
    cats = DATASET_VERSIONS[version]
    if split == "train":
        return generate(800 // len(cats) * len(cats), seed=7, categories=cats)
    if split == "test":
        return generate(126 // len(cats) * len(cats), seed=11, extra=TEST_ONLY_TEMPLATES, categories=cats)
    raise ValueError(f"unknown split {split!r}; use train or test")


def as_chat(ticket: Ticket, with_answer: bool, version: str = "v1") -> list[dict[str, str]]:
    """A ticket as a chat transcript, for both training and inference."""
    msgs = [{"role": "system", "content": system_prompt(version)}, {"role": "user", "content": ticket.text}]
    if with_answer:
        msgs.append({"role": "assistant", "content": ticket.label})
    return msgs


def parse_label(text: str, categories: list[str] = CATEGORIES) -> str:
    """The first category name found in a model's reply, else 'other'."""
    low = text.lower()
    best, best_pos = "other", len(low) + 1
    for cat in categories:
        pos = low.find(cat)
        if pos != -1 and pos < best_pos:
            best, best_pos = cat, pos
    return best
