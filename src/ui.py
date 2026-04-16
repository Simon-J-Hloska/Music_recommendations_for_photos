"""
image_drop.py — Minimal drag-and-drop image filepath collector

Usage:
    drop = ImageDrop()
    drop.run()                     # blocks; drop a file to capture its path
    print(drop.filepath)           # the dropped file's path

    drop.display("hello world")    # print anything into the tkinter window
"""
import ctypes
ctypes.WinDLL("shcore").SetProcessDpiAwareness(1)
import shutil
import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD

class ImageDrop:
    def __init__(self):
        self.filepath_list = None
        self._root = TkinterDnD.Tk()
        self._root.title("Drop Image")
        self._root.configure(bg="#1a1a1a")
        self._root.geometry("800x800")
        self.counter = 0
        self._done = False
        self.button_frame = None
        self._label = tk.Label(
            self._root,
            text="drop one or more images here",
            bg="#1a1a1a", fg="#666666",
            font=("Courier", 13),
            wraplength=280, justify="center"
        )
        self._label.pack(expand=True)
        self._text = tk.Text(
            self._root,
            bg="#1a1a1a",
            fg="#cccccc",
            font=("Courier", 11),
            wrap="word",
            state="disabled"
        )
        self._text.pack(expand=True, fill="both")
        self._root.drop_target_register(DND_FILES)
        self._root.dnd_bind("<<Drop>>", self._on_drop)

    def _on_drop(self, event):
        files = self._root.tk.splitlist(event.data)
        self.filepath_list = list(files)
        self._label.config(fg="#cccccc", text="Images dropped, proceeding to processing...")
        self.save_image()
        self._done = True
        self._root.quit()  # stops mainloop

    def clear(self):
        self._label.config(text="")

    def display(self, text: str):
        self._text.config(state="normal")
        self._text.insert("end", str(text) + "\n")
        self._text.see("end")  # auto-scroll
        self._text.config(state="disabled")

    def run(self):
        self._root.mainloop()

    def show_action_buttons(self):
        """Show Restart and Close buttons after results are displayed."""
        self._action_taken = False
        self.btn_frame = tk.Frame(self._root, bg="#1a1a1a")
        self.btn_frame.pack(pady=10)
        def on_restart():
            self._action_taken = "restart"
            self._root.quit()
        def on_close():
            self._action_taken = "close"
            self._root.quit()
        tk.Button(
            self.btn_frame, text="Restart",
            bg="#2a2a2a", fg="#cccccc",
            font=("Courier", 11), width=12,
            command=on_restart
        ).pack(side="left", padx=10)
        tk.Button(
            self.btn_frame, text="Close",
            bg="#2a2a2a", fg="#cccccc",
            font=("Courier", 11), width=12,
            command=on_close
        ).pack(side="left", padx=10)
        self._root.mainloop()
        return self._action_taken

    def show_results(self, matches, image_name):
        self.display(f"\nFor image: {image_name}, here are the stats:")
        for i, m in enumerate(matches, 1):
            self.display(f"{i}. {m['track_name']} — {m['artists']}")

    def save_image(self):
        for image in self.filepath_list:
            self.counter += 1
            shutil.copy(image, f"../data/images/user_images/raw/user_image{self.counter}.jpg")

    def reset(self):
        self._text.config(state="normal")
        self._text.delete("1.0", "end")
        self._text.config(state="disabled")
        self._label.config(
            text="drop one or more images here",
            fg="#666666")
        if self.btn_frame:
            self.btn_frame.destroy()