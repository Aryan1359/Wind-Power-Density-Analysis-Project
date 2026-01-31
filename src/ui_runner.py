import os
import shutil
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")

# Make sure folders exist
os.makedirs(DATA_RAW, exist_ok=True)

# Global for legend selection
legend_box = None  # (x1, y1, x2, y2)
start_x = start_y = None
rect_id = None


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Wind Project UI (Module 5) - Pick Image → Select Legend → Run")
        self.geometry("1100x700")

        self.image_path = None
        self.last_figure_path = None
        self.last_csv_path = None
        self.img = None
        self.tk_img = None
        self.canvas = tk.Canvas(self, bg="white", width=900, height=550)
        self.canvas.pack(pady=10)

        # Buttons
        frame = tk.Frame(self)
        frame.pack()

        tk.Button(frame, text="1) Select Screenshot", command=self.select_image, width=20).grid(row=0, column=0, padx=6)
        tk.Button(frame, text="2) Select Legend (drag box)", command=self.enable_legend_select, width=24).grid(row=0, column=1, padx=6)
        tk.Button(frame, text="3) Run Analysis", command=self.run_analysis, width=20).grid(row=0, column=2, padx=6)
        tk.Button(frame, text="Open Output Figure", command=self.open_output, width=20).grid(row=0, column=3, padx=6)

        # Settings
        settings = tk.Frame(self)
        settings.pack(pady=8)

        tk.Label(settings, text="mph min (bottom):").grid(row=0, column=0, sticky="e")
        self.mph_min_var = tk.StringVar(value="0")
        tk.Entry(settings, textvariable=self.mph_min_var, width=8).grid(row=0, column=1, padx=6)

        tk.Label(settings, text="mph max (top):").grid(row=0, column=2, sticky="e")
        self.mph_max_var = tk.StringVar(value="40")
        tk.Entry(settings, textvariable=self.mph_max_var, width=8).grid(row=0, column=3, padx=6)

        tk.Label(settings, text="grid size:").grid(row=0, column=4, sticky="e")
        self.grid_var = tk.StringVar(value="20")
        tk.Entry(settings, textvariable=self.grid_var, width=8).grid(row=0, column=5, padx=6)

        self.status = tk.Label(self, text="Step 1: Select your wind map screenshot.", fg="blue")
        self.status.pack(pady=6)

        # Mouse bindings for selection (disabled until enabled)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.legend_select_enabled = False

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Select your Zoom Earth wind screenshot",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp")]
        )
        if not path:
            return

        # Copy into data/raw/ preserving original base name
        base_name = os.path.basename(path)
        dest_path = os.path.join(DATA_RAW, base_name)
        try:
            if os.path.abspath(path) != os.path.abspath(dest_path):
                shutil.copy(path, dest_path)
        except OSError as exc:
            messagebox.showerror("Copy failed", f"Could not copy image to data/raw/: {exc}")
            return

        self.image_path = dest_path
        self.status.config(
            text=(
                f"Saved screenshot to: {self.image_path}\n"
                "Now click: 'Select Legend (drag box)'."
            ),
            fg="green",
        )
        self.load_and_show_image()

    def load_and_show_image(self):
        self.img = Image.open(self.image_path).convert("RGB")

        # Resize only for display (keep original for processing)
        disp = self.img.copy()
        disp.thumbnail((900, 550))
        self.display_scale = (self.img.size[0] / disp.size[0], self.img.size[1] / disp.size[1])

        self.tk_img = ImageTk.PhotoImage(disp)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.canvas.config(width=disp.size[0], height=disp.size[1])

    def enable_legend_select(self):
        if not self.image_path:
            messagebox.showwarning("No image", "Please select a screenshot first.")
            return
        self.legend_select_enabled = True
        self.status.config(text="Drag a rectangle around the legend colorbar (ONLY the colored bar).", fg="blue")

    def on_mouse_down(self, event):
        global start_x, start_y, rect_id, legend_box
        if not self.legend_select_enabled:
            return
        start_x, start_y = event.x, event.y
        if rect_id:
            self.canvas.delete(rect_id)
            rect_id = None
        legend_box = None

    def on_mouse_drag(self, event):
        global rect_id
        if not self.legend_select_enabled or start_x is None:
            return
        if rect_id:
            self.canvas.delete(rect_id)
        rect_id = self.canvas.create_rectangle(start_x, start_y, event.x, event.y, outline="red", width=2)

    def on_mouse_up(self, event):
        global legend_box
        if not self.legend_select_enabled:
            return

        x1, y1 = start_x, start_y
        x2, y2 = event.x, event.y

        # Normalize
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        # Convert display coords -> original image coords
        sx, sy = self.display_scale
        ox1, oy1 = int(x1 * sx), int(y1 * sy)
        ox2, oy2 = int(x2 * sx), int(y2 * sy)

        legend_box = (ox1, oy1, ox2, oy2)
        self.legend_select_enabled = False
        self.status.config(text=f"Legend selected: {legend_box}\nNow click: 'Run Analysis'.", fg="green")

    def run_analysis(self):
        global legend_box
        if not self.image_path:
            messagebox.showwarning("No image", "Please select a screenshot first.")
            return
        if not legend_box:
            messagebox.showwarning("No legend", "Please select the legend colorbar (drag a box) first.")
            return

        try:
            mph_min = float(self.mph_min_var.get())
            mph_max = float(self.mph_max_var.get())
            grid_size = int(self.grid_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter valid numbers for mph min/max and grid size.")
            return

        # Build command to run main_analysis.py
        main_script = os.path.join(PROJECT_ROOT, "src", "main_analysis.py")
        cmd = [
            sys.executable, main_script,
            "--image", self.image_path,
            "--legend", str(legend_box[0]), str(legend_box[1]), str(legend_box[2]), str(legend_box[3]),
            "--mph-min", str(mph_min),
            "--mph-max", str(mph_max),
            "--grid-size", str(grid_size),
        ]

        self.status.config(text="Running analysis... please wait.", fg="blue")
        self.update_idletasks()

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        except OSError as exc:
            messagebox.showerror("Run failed", f"Failed to start analysis process: {exc}")
            return

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        figure_path = None
        csv_path = None
        for line in stdout.splitlines():
            if line.startswith("FIGURE_OUTPUT:"):
                figure_path = line.split(":", 1)[1].strip()
            elif line.startswith("CSV_OUTPUT:"):
                csv_path = line.split(":", 1)[1].strip()

        if result.returncode != 0:
            message = "Analysis failed. Check VS Code terminal output for details."
            if stderr.strip():
                message += f"\n\nError details:\n{stderr.strip()}"
            messagebox.showerror("Run failed", message)
            self.status.config(text="Analysis failed.", fg="red")
            return

        self.last_figure_path = figure_path
        self.last_csv_path = csv_path

        status_lines = ["Done!"]
        if figure_path:
            status_lines.append(f"Figure: {figure_path}")
        if csv_path:
            status_lines.append(f"CSV: {csv_path}")

        self.status.config(text="\n".join(status_lines), fg="green")
        self.open_output()

    def open_output(self):
        path = self.last_figure_path
        if path and os.path.exists(path):
            os.startfile(path)  # Windows
        else:
            messagebox.showinfo(
                "Not found",
                "Output figure not found yet. Run analysis first.",
            )


if __name__ == "__main__":
    app = App()
    app.mainloop()
