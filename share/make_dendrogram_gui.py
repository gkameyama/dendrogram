from __future__ import annotations

from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from make_dendrogram import SCRIPT_DIR, create_dendrogram


class DendrogramApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Dendrogram Maker")
        self.geometry("720x430")
        self.minsize(640, 360)

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.matrix_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()

        self._build_ui()
        self.after(100, self._poll_messages)

    def _build_ui(self) -> None:
        frame = ttk.Frame(self, padding=16)
        frame.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(4, weight=1)

        ttk.Label(frame, text="Input Excel").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.input_var).grid(
            row=0, column=1, sticky="ew", padx=8, pady=4
        )
        ttk.Button(frame, text="Browse", command=self._browse_input).grid(
            row=0, column=2, pady=4
        )

        ttk.Label(frame, text="Output PNG").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.output_var).grid(
            row=1, column=1, sticky="ew", padx=8, pady=4
        )
        ttk.Button(frame, text="Save As", command=self._browse_output).grid(
            row=1, column=2, pady=4
        )

        ttk.Label(frame, text="Distance Matrix").grid(
            row=2, column=0, sticky="w", pady=4
        )
        ttk.Entry(frame, textvariable=self.matrix_var).grid(
            row=2, column=1, sticky="ew", padx=8, pady=4
        )
        ttk.Button(frame, text="Save As", command=self._browse_matrix).grid(
            row=2, column=2, pady=4
        )

        controls = ttk.Frame(frame)
        controls.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(12, 8))
        controls.columnconfigure(1, weight=1)
        self.run_button = ttk.Button(
            controls, text="Create Dendrogram", command=self._run
        )
        self.run_button.grid(row=0, column=0, sticky="w")
        ttk.Label(controls, textvariable=self.status_var).grid(
            row=0, column=1, sticky="e"
        )

        self.log = tk.Text(frame, height=10, wrap="word", state="disabled")
        self.log.grid(row=4, column=0, columnspan=3, sticky="nsew")

    def _browse_input(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=SCRIPT_DIR,
            title="Select input Excel file",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
        )
        if path:
            self.input_var.set(path)

    def _browse_output(self) -> None:
        path = filedialog.asksaveasfilename(
            initialdir=SCRIPT_DIR,
            title="Select output PNG file",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
        )
        if path:
            self.output_var.set(path)

    def _browse_matrix(self) -> None:
        path = filedialog.asksaveasfilename(
            initialdir=SCRIPT_DIR,
            title="Select distance matrix Excel file",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
        )
        if path:
            self.matrix_var.set(path)

    def _run(self) -> None:
        input_path = self._optional_path(self.input_var.get())
        output_path = self._optional_path(self.output_var.get())
        matrix_path = self._optional_path(self.matrix_var.get())

        self.run_button.configure(state="disabled")
        self.status_var.set("Running...")
        self._append_log("Creating dendrogram...\n")

        thread = threading.Thread(
            target=self._worker,
            args=(input_path, output_path, matrix_path),
            daemon=True,
        )
        thread.start()

    def _worker(
        self,
        input_path: Path | None,
        output_path: Path | None,
        matrix_path: Path | None,
    ) -> None:
        try:
            result = create_dendrogram(input_path, output_path, matrix_path)
        except Exception as exc:
            self.messages.put(("error", exc))
        else:
            self.messages.put(("done", result))

    def _poll_messages(self) -> None:
        try:
            kind, payload = self.messages.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_messages)
            return

        self.run_button.configure(state="normal")
        if kind == "error":
            self.status_var.set("Failed")
            self._append_log(f"Error: {payload}\n")
            messagebox.showerror("Dendrogram Maker", str(payload))
        else:
            result = payload
            self.status_var.set("Done")
            self._append_log(
                "Done.\n"
                f"Input  : {result['input']}\n"
                f"Data   : {result['data_sheet']}\n"
                f"Label  : {result['label_sheet']}\n"
                f"Matrix : {result['matrix']}\n"
                f"Saved  : {result['saved']}\n"
            )
            messagebox.showinfo("Dendrogram Maker", "Dendrogram was created.")

        self.after(100, self._poll_messages)

    def _append_log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message)
        self.log.see("end")
        self.log.configure(state="disabled")

    @staticmethod
    def _optional_path(value: str) -> Path | None:
        stripped = value.strip().strip('"')
        return Path(stripped) if stripped else None


if __name__ == "__main__":
    DendrogramApp().mainloop()
