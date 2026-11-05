"""Toy Acme ERP desktop client: Price Maintenance screen (rung 6, no API).

Deliberately a little hostile to recorded-click bots: a confirmation dialog for changes over
10%, and an occasional "session timeout" warning that has to be dismissed before Find works.
"""

from __future__ import annotations

import os
import random
import tkinter as tk
from tkinter import messagebox, ttk

import acme_db

SURPRISE_RATE = float(os.environ.get("ACME_SURPRISE_RATE", "0.15"))
REASONS = ["Supplier cost update", "Promotion", "Correction", "Competitive match"]


class PriceMaintenance(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        acme_db.init()
        self.title("Acme ERP 7.2 - Price Maintenance")
        self.geometry("900x620+150+60")
        self.configure(bg="#dfe3e8")
        self.option_add("*Font", "DejaVuSans 11")
        self.current: dict | None = None

        bar = tk.Frame(self, bg="#1f3a5f", height=36)
        bar.pack(fill="x")
        tk.Label(bar, text="ACME ERP  |  Pricing  >  Price Maintenance", fg="white", bg="#1f3a5f",
                 font=("DejaVuSans", 12, "bold")).pack(side="left", padx=10, pady=6)

        form = tk.LabelFrame(self, text=" Item lookup ", bg="#dfe3e8", padx=10, pady=8)
        form.pack(fill="x", padx=12, pady=10)
        tk.Label(form, text="SKU:", bg="#dfe3e8").grid(row=0, column=0, sticky="w")
        self.sku_var = tk.StringVar()
        self.sku_entry = tk.Entry(form, textvariable=self.sku_var, width=20)
        self.sku_entry.grid(row=0, column=1, padx=6)
        self.sku_entry.bind("<Return>", lambda _e: self.find())
        tk.Button(form, text="Find", width=10, command=self.find).grid(row=0, column=2, padx=6)

        tk.Label(form, text="Description:", bg="#dfe3e8").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.desc = tk.Label(form, text="-", bg="#dfe3e8", anchor="w", width=60)
        self.desc.grid(row=1, column=1, columnspan=3, sticky="w", pady=(8, 0))
        tk.Label(form, text="Current price:", bg="#dfe3e8").grid(row=2, column=0, sticky="w")
        self.cur = tk.Label(form, text="-", bg="#dfe3e8", anchor="w")
        self.cur.grid(row=2, column=1, sticky="w")

        edit = tk.LabelFrame(self, text=" Price override ", bg="#dfe3e8", padx=10, pady=8)
        edit.pack(fill="x", padx=12)
        tk.Label(edit, text="New price ($):", bg="#dfe3e8").grid(row=0, column=0, sticky="w")
        self.new_var = tk.StringVar()
        tk.Entry(edit, textvariable=self.new_var, width=12).grid(row=0, column=1, padx=6, sticky="w")
        tk.Label(edit, text="Reason:", bg="#dfe3e8").grid(row=0, column=2, sticky="w", padx=(20, 0))
        self.reason = ttk.Combobox(edit, values=REASONS, state="readonly", width=24)
        self.reason.current(0)
        self.reason.grid(row=0, column=3, padx=6)
        self.save_btn = tk.Button(edit, text="Save", width=10, state="disabled", command=self.save)
        self.save_btn.grid(row=0, column=4, padx=10)

        hist = tk.LabelFrame(self, text=" Recent changes ", bg="#dfe3e8", padx=6, pady=6)
        hist.pack(fill="both", expand=True, padx=12, pady=10)
        cols = ("ts", "sku", "old", "new", "source")
        self.tree = ttk.Treeview(hist, columns=cols, show="headings", height=10)
        for c, w in zip(cols, (200, 140, 90, 90, 120)):
            self.tree.heading(c, text=c.upper())
            self.tree.column(c, width=w)
        self.tree.pack(fill="both", expand=True)

        self.status = tk.Label(self, text="Ready", anchor="w", bg="#c9ced6")
        self.status.pack(fill="x", side="bottom")
        self.refresh_history()
        self.sku_entry.focus_set()

    def find(self) -> None:
        if random.random() < SURPRISE_RATE:
            messagebox.showwarning("Session", "Your session will expire in 5 minutes.\nClick OK to stay signed in.")
        row = acme_db.get_sku(self.sku_var.get())
        if row is None:
            self.current = None
            self.desc.config(text="-")
            self.cur.config(text="-")
            self.save_btn.config(state="disabled")
            self.status.config(text=f"SKU '{self.sku_var.get()}' not found")
            return
        self.current = row
        self.desc.config(text=row["description"])
        self.cur.config(text=f"${row['price']:.2f}")
        self.new_var.set("")
        self.save_btn.config(state="normal")
        self.status.config(text=f"Loaded {row['sku']}")

    def save(self) -> None:
        if not self.current:
            return
        try:
            new = float(self.new_var.get().replace("$", "").strip())
        except ValueError:
            messagebox.showerror("Invalid price", "Enter a numeric price, e.g. 42.50")
            return
        old = self.current["price"]
        if old and abs(new - old) / old > 0.10:
            if not messagebox.askyesno(
                "Confirm large change",
                f"{self.current['sku']}: price change of {100 * (new - old) / old:+.1f}% exceeds 10%.\n\nApply anyway?",
            ):
                self.status.config(text="Change cancelled")
                return
        res = acme_db.set_price(self.current["sku"], new, self.reason.get(), "desktop")
        self.current["price"] = res["new_price"]
        self.cur.config(text=f"${res['new_price']:.2f}")
        self.status.config(text=f"Saved {res['sku']}: ${res['old_price']:.2f} -> ${res['new_price']:.2f}")
        self.refresh_history()

    def refresh_history(self) -> None:
        self.tree.delete(*self.tree.get_children())
        for a in reversed(acme_db.audit()[-25:]):
            self.tree.insert("", "end", values=(a["ts"], a["sku"], f"{a['old_price']:.2f}", f"{a['new_price']:.2f}", a["source"]))


if __name__ == "__main__":
    PriceMaintenance().mainloop()
