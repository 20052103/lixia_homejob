# -*- coding: utf-8 -*-
"""
Contact list manager.

Contacts are stored in contacts.json (same directory).
VIP contacts are always checked during email fetches and will be
the foundation for the future send-email feature.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

_DEFAULT_CONTACTS_PATH = os.path.join(os.path.dirname(__file__), "contacts.json")


@dataclass
class Contact:
    name: str
    email: str

    def __str__(self) -> str:
        return f"{self.name} <{self.email}>"


def load_vip_contacts(path: str = _DEFAULT_CONTACTS_PATH) -> List[Contact]:
    """Load VIP contacts from contacts.json."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        Contact(name=c["name"], email=c["email"])
        for c in data.get("vip_contacts", [])
    ]


def _save_contacts(contacts: List[Contact], path: str) -> None:
    """Persist the contacts list to JSON."""
    data = {"vip_contacts": [{"name": c.name, "email": c.email} for c in contacts]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def deduplicate_contacts(path: str = _DEFAULT_CONTACTS_PATH) -> Tuple[List[Contact], List[Contact]]:
    """
    Remove duplicate entries from contacts.json.

    Duplicates are defined as entries sharing the same email address
    (case-insensitive). The first occurrence is kept.

    Returns (kept, removed) lists.
    """
    existing = load_vip_contacts(path)
    seen: dict = {}
    kept: List[Contact] = []
    removed: List[Contact] = []
    for c in existing:
        key = c.email.strip().lower()
        if key in seen:
            removed.append(c)
        else:
            seen[key] = True
            kept.append(c)
    if removed:
        _save_contacts(kept, path)
    return kept, removed


def add_vip_contact(name: str, email: str, path: str = _DEFAULT_CONTACTS_PATH) -> Contact:
    """
    Add a new VIP contact and persist to contacts.json.
    Raises ValueError if the email already exists (case-insensitive).
    """
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"vip_contacts": []}

    existing_emails = {c["email"].strip().lower() for c in data["vip_contacts"]}
    if email.strip().lower() in existing_emails:
        raise ValueError(f"Contact with email {email!r} already exists.")

    data["vip_contacts"].append({"name": name, "email": email})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return Contact(name=name, email=email)


def build_vip_gmail_query(contacts: List[Contact]) -> Optional[str]:
    """Build a Gmail search query that matches any VIP contact as sender."""
    if not contacts:
        return None
    parts = [f"from:{c.email}" for c in contacts]
    return " OR ".join(parts)


def is_from_vip(sender: str, contacts: List[Contact]) -> Optional[Contact]:
    """Return the matching VIP Contact if the sender is a VIP, else None."""
    sender_lower = sender.lower()
    for c in contacts:
        if c.email.lower() in sender_lower:
            return c
    return None
