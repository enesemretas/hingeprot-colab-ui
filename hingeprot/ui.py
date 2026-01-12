from __future__ import annotations

import os, re, shutil, subprocess, datetime, base64, uuid
import requests
import ipywidgets as W
from IPython.display import display, clear_output


def launch(runs_root: str = "/content/hingeprot_runs"):
    """
    Launch the UI in the current notebook kernel.
    IMPORTANT: call this from a notebook cell (not via python3 ui.py).
    """
    from google.colab import output  # colab-only
    output.enable_custom_widget_manager()

    HINGEPROT_DIR = os.path.dirname(os.path.abspath(__file__))
    READ_PY = os.path.join(HINGEPROT_DIR, "read.py")

    os.makedirs(runs_root, exist_ok=True)

    # ---------- helpers ----------
    def _fetch_pdb(pdb_code: str, out_path: str):
        code = pdb_code.strip().upper()
        if not re.fullmatch(r"[0-9A-Z]{4}", code):
            raise ValueError("PDB code must be 4 characters (e.g., 4CLN).")
        url = f"https://files.rcsb.org/download/{code}.pdb"
        r = requests.get(url, timeout=30)
        if r.status_code != 200 or len(r.text) < 200:
            raise RuntimeError(f"Failed to fetch PDB {code} (HTTP {r.status_code}).")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(r.text)

    def _detect_chains(pdb_path: str):
        chains = set()
        with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith(("ATOM  ", "HETATM")) and len(line) > 21:
                    ch = line[21].strip()
                    if ch:
                        chains.add(ch)
        return sorted(chains)

    def _write_text(path: str, text: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(text).strip() + "\n")

    def _preprocess_pdb(
        pdb_in: str,
        pdb_out: str,
        chains_keep: set[str] | None,
        keep_ter: bool = True,
        keep_only_first_model: bool = True,
    ) -> dict:
        """
        Preprocess (MATLAB logic adapted to raw PDB text):

        - Keep ONLY first MODEL block (if present) else all.
        - Keep only ATOM (and optionally TER). Drop HETATM and everything else.
        - altLoc: keep if blank or 'A'
        - iCode : keep if blank
        - chain : keep only selected chains (if chains_keep is not None)

        Returns stats dict.
        """
        stats = {
            "lines_read": 0,
            "lines_written": 0,
            "atom_written": 0,
            "ter_written": 0,
            "altloc_skipped": 0,
            "icode_skipped": 0,
            "chain_skipped": 0,
            "hetatm_skipped": 0,
            "nonatom_skipped": 0,
            "used_model_block": False,
        }

        in_model = False
        model_found = False
        model_done = False

        with open(pdb_in, "r", encoding="utf-8", errors="ignore") as fin, \
             open(pdb_out, "w", encoding="utf-8") as fout:

            for line in fin:
                stats["lines_read"] += 1
                rec = line[:6]

                # MODEL logic: keep only first MODEL block (if present)
                if keep_only_first_model:
                    if rec == "MODEL ":
                        if not model_found:
                            model_found = True
                            in_model = True
                            stats["used_model_block"] = True
                        else:
                            in_model = False
                            model_done = True
                        continue

                    if rec == "ENDMDL":
                        if model_found and in_model:
                            in_model = False
                            model_done = True
                        continue

                    if model_done:
                        continue

                    # If MODEL exists, ignore anything outside the first MODEL
                    if model_found and not in_model:
                        continue

                # Drop HETATM
                if rec == "HETATM":
                    stats["hetatm_skipped"] += 1
                    continue

                # TER optional
                if rec == "TER   ":
                    if not keep_ter:
                        stats["nonatom_skipped"] += 1
                        continue
                    if chains_keep is not None and len(line) > 21:
                        ch = line[21]
                        if ch not in chains_keep:
                            stats["chain_skipped"] += 1
                            continue
                    fout.write(line if line.endswith("\n") else line + "\n")
                    stats["lines_written"] += 1
                    stats["ter_written"] += 1
                    continue

                # Keep only ATOM
                if rec != "ATOM  ":
                    stats["nonatom_skipped"] += 1
                    continue

                # chain filter
                if chains_keep is not None:
                    if len(line) <= 21:
                        stats["chain_skipped"] += 1
                        continue
                    ch = line[21]
                    if ch not in chains_keep:
                        stats["chain_skipped"] += 1
                        continue

                # altLoc filter (col 17 => index 16)
                altloc = line[16] if len(line) > 16 else " "
                if altloc not in (" ", "A"):
                    stats["altloc_skipped"] += 1
                    continue

                # iCode filter (col 27 => index 26)
                icode = line[26] if len(line) > 26 else " "
                if icode != " ":
                    stats["icode_skipped"] += 1
                    continue

                fout.write(line if line.endswith("\n") else line + "\n")
                stats["lines_written"] += 1
                stats["atom_written"] += 1

        return stats

    # ---------- UI helpers ----------
    def _list_or_custom_float(label: str, options, default_value: float,
                              minv: float, maxv: float, step: float = 0.1,
                              label_width="120px", toggle_width="180px", value_width="240px"):
        opts = [float(x) for x in options]
        default_value = float(default_value)
        if default_value not in opts:
            opts = sorted(set(opts + [default_value]))
        else:
            opts = sorted(set(opts))

        lbl = W.Label(label, layout=W.Layout(width=label_width))
        toggle = W.ToggleButtons(
            options=[("List", "list"), ("Custom", "custom")],
            value="list",
            layout=W.Layout(width=toggle_width),
            style={"button_width": "80px"}
        )
        dropdown = W.Dropdown(options=opts, value=default_value, layout=W.Layout(width=value_width))
        fbox = W.BoundedFloatText(value=default_value, min=minv, max=maxv, step=step, layout=W.Layout(width=value_width))
        value_box = W.Box([dropdown], layout=W.Layout(align_items="center"))

        def _on_toggle(ch):
            value_box.children = [dropdown] if ch["new"] == "list" else [fbox]
        toggle.observe(_on_toggle, names="value")

        def get_value() -> float:
            return float(dropdown.value) if toggle.value == "list" else float(fbox.value)

        row = W.HBox([lbl, toggle, value_box], layout=W.Layout(align_items="center", gap="12px"))
        return row, get_value

    # ---------- UI ----------
    css = W.HTML(r"""
    <style>
    .hp-card {border:1px solid #e5e7eb; border-radius:14px; padding:14px 16px; margin:10px 0; background:#fff;}
    .hp-small {font-size:12px; color:#6b7280; margin-top:6px;}

    /* Banner like your logo style (text-only, no image) */
    .hp-banner{
      border:1px solid #e5e7eb;
      border-radius:16px;
      padding:14px 18px;
      margin:10px 0 12px 0;
      background:#fff;
      display:flex;
      align-items:center;
      gap:16px;
      box-shadow: 0 1px 0 rgba(0,0,0,0.03);
    }
    .hp-dot{
      width:14px; height:14px;
      background:#ef4444;
      border-radius:999px;
      flex:0 0 auto;
    }
    .hp-title{
      font-size:34px;
      font-weight:900;
      letter-spacing:0.5px;
      line-height:1.0;
      margin:0;
      color:#111827;
      font-family: Arial, Helvetica, sans-serif;
    }
    .hp-title .prot{ color:#ef4444; }
    .hp-underline{
      height:3px;
      width:280px;
      background:#111827;
      margin-top:6px;
      border-radius:999px;
      opacity:0.9;
    }
    .hp-tagline{
      margin-top:6px;
      font-size:16px;
      font-weight:800;
      color:#dc2626;
      font-family: Arial, Helvetica, sans-serif;
    }
    </style>
    """)

    header = W.HTML(r"""
    <div class="hp-banner">
      <div class="hp-dot"></div>
      <div>
        <div class="hp-title">HINGE<span class="prot">prot</span></div>
        <div class="hp-underline"></div>
        <div class="hp-tagline">An Algorithm For Protein Hinge Prediction Using Elastic Network Models</div>
      </div>
    </div>
    """)

    input_mode = W.RadioButtons(
        options=[("Enter PDB code", "code"), ("Upload PDB file", "upload")],
        value="code",
        description="Input:",
        style={"description_width": "60px"},
        layout=W.Layout(width="420px")
    )

    pdb_code = W.Text(
        value="",
        description="PDB code:",
        placeholder="e.g., 4cln",
        style={"description_width": "80px"},
        layout=W.Layout(width="420px")
    )

    btn_choose_file = W.Button(description="Choose file", icon="upload", layout=W.Layout(width="180px"))
    file_lbl = W.Label("No file chosen")

    code_box = W.HBox([pdb_code], layout=W.Layout(align_items="center"))
    upload_box = W.HBox([btn_choose_file, file_lbl], layout=W.Layout(align_items="center", gap="10px"))

    btn_load = W.Button(description="Load / Detect Chains", button_style="info", icon="search",
                        layout=W.Layout(width="260px"))

    chains_select = W.SelectMultiple(
        options=[], description="Select Chains:", rows=8,
        style={"description_width":"110px"},
        layout=W.Layout(width="420px")
    )
    all_chains = W.Checkbox(value=False, description="All chains")

    gnm_row, get_gnm_cut = _list_or_custom_float(
        "GNM cutoff (Å):", options=[7,8,9,10,11,12,13,20], default_value=10.0, minv=1.0, maxv=100.0
    )
    anm_row, get_anm_cut = _list_or_custom_float(
        "ANM cutoff (Å):", options=[10,13,15,18,20,23,36], default_value=18.0, minv=1.0, maxv=100.0
    )
    rescale = W.BoundedFloatText(
        value=1.0, min=0.01, max=100.0, step=0.01,
        description="Rescale:",
        style={"description_width":"80px"},
        layout=W.Layout(width="260px")
    )

    progress = W.IntProgress(value=0, min=0, max=1, description="Progress:", bar_style="")
    btn_run  = W.Button(description="Preprocess + run read.py", button_style="success", icon="play",
                        layout=W.Layout(width="320px"))
    btn_clear= W.Button(description="Clear", button_style="warning", icon="trash", layout=W.Layout(width="180px"))

    log_out = W.Output()

    state = {
        "pdb_path": None,
        "run_dir": None,
        "upload_name": None,
        "upload_bytes": None,
        "detected_chains": [],
    }

    def _show_log(msg: str):
        with log_out:
            print(msg)

    def _sync_input_visibility(*_):
        if input_mode.value == "code":
            code_box.layout.display = ""
            upload_box.layout.display = "none"
        else:
            code_box.layout.display = "none"
            upload_box.layout.display = ""
    _sync_input_visibility()
    input_mode.observe(lambda ch: _sync_input_visibility(), names="value")

    # ---- One-click uploader: unique callback name to avoid collisions ----
    cb_name = f"hingeprot_uploader_{uuid.uuid4().hex}"

    def _js_upload_callback(payload):
        try:
            name = payload.get("name", "upload.pdb")
            data_b64 = payload.get("data_b64", "")
            if not data_b64:
                _show_log("Upload callback received empty data.")
                return
            data = base64.b64decode(data_b64.encode("utf-8"))
            state["upload_name"] = name
            state["upload_bytes"] = data
            file_lbl.value = name
            _show_log(f"Uploaded file received: {name} ({len(data)} bytes)")
        except Exception as e:
            _show_log(f"Upload callback error: {e}")

    output.register_callback(cb_name, _js_upload_callback)

    def on_choose_file(_):
        js = f"""
        (async () => {{
          const input = document.createElement('input');
          input.type = 'file';
          input.accept = '.pdb,.ent';
          input.style.display = 'none';
          document.body.appendChild(input);

          input.onchange = async () => {{
            const file = input.files && input.files[0];
            document.body.removeChild(input);
            if (!file) return;

            const reader = new FileReader();
            reader.onload = async () => {{
              const b64 = (reader.result || "").split(",")[1] || "";
              await google.colab.kernel.invokeFunction(
                "{cb_name}",
                [{{name: file.name, data_b64: b64}}],
                {{}}
              );
            }};
            reader.readAsDataURL(file);
          }};

          input.click();
        }})();
        """
        output.eval_js(js)

    btn_choose_file.on_click(on_choose_file)

    def on_load_clicked(_):
        with log_out:
            clear_output()

        try:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(runs_root, f"run_{ts}")
            os.makedirs(run_dir, exist_ok=True)
            state["run_dir"] = run_dir

            pdb_path = os.path.join(run_dir, "input.pdb")

            if input_mode.value == "upload":
                if state["upload_bytes"] is None:
                    raise ValueError("Please click 'Choose file' and upload a PDB first.")
                _show_log(f"Saving uploaded PDB: {state['upload_name']}")
                with open(pdb_path, "wb") as f:
                    f.write(state["upload_bytes"])
            else:
                code = pdb_code.value.strip()
                if not code:
                    raise ValueError("Please enter a PDB code (e.g., 4cln).")
                _show_log(f"Downloading PDB {code} ...")
                _fetch_pdb(code, pdb_path)

            state["pdb_path"] = pdb_path
            _show_log(f"Input PDB saved: {pdb_path} (size={os.path.getsize(pdb_path)} bytes)")

            chs = _detect_chains(pdb_path)
            if not chs:
                raise RuntimeError("No chains detected in the PDB.")
            state["detected_chains"] = chs
            chains_select.options = chs
            chains_select.value = tuple(chs[:1])

            _show_log(f"Detected chains: {chs}")
            _show_log("Select chain(s) or tick 'All chains', then Run.")
        except Exception as e:
            _show_log(f"ERROR: {e}")

    def on_run_clicked(_):
        with log_out:
            clear_output()

        try:
            if not state["pdb_path"] or not os.path.exists(state["pdb_path"]):
                raise RuntimeError("Please click 'Load / Detect Chains' first.")

            if not os.path.exists(READ_PY):
                raise RuntimeError(f"read.py not found at: {READ_PY}")

            detected = state.get("detected_chains") or list(chains_select.options)
            if not detected:
                raise RuntimeError("No detected chains. Load again.")

            if all_chains.value:
                chain_list = detected
            else:
                chain_list = list(chains_select.value)
                if not chain_list:
                    raise RuntimeError("Please select at least one chain (or tick All chains).")

            chain_str = "".join(chain_list)
            chains_keep = None if all_chains.value else set(chain_str)

            gnm_val = float(get_gnm_cut())
            anm_val = float(get_anm_cut())
            rescale_val = float(rescale.value)

            progress.max = 5
            progress.value = 0
            progress.bar_style = "info"

            _show_log(f"Run folder: {state['run_dir']}")
            _show_log(f"Selected chains: {(chain_str if not all_chains.value else 'ALL')}")

            # Write parameters into run folder (for now)
            _write_text(os.path.join(state["run_dir"], "gnmcutoff"), gnm_val)
            _write_text(os.path.join(state["run_dir"], "anmcutoff"), anm_val)
            _write_text(os.path.join(state["run_dir"], "rescale"), rescale_val)
            progress.value += 1
            _show_log("Parameters written: gnmcutoff / anmcutoff / rescale")

            # Preprocess into run_dir/pdb
            pdb_out = os.path.join(state["run_dir"], "pdb")
            stats = _preprocess_pdb(
                pdb_in=state["pdb_path"],
                pdb_out=pdb_out,
                chains_keep=chains_keep,
                keep_ter=True,
                keep_only_first_model=True,
            )
            progress.value += 1
            _show_log("Preprocess done. Output written as 'pdb'.")
            _show_log(
                f"Stats: atom_written={stats['atom_written']}, ter_written={stats['ter_written']}, "
                f"altloc_skipped={stats['altloc_skipped']}, icode_skipped={stats['icode_skipped']}, "
                f"chain_skipped={stats['chain_skipped']}, hetatm_skipped={stats['hetatm_skipped']}, "
                f"used_model_block={stats['used_model_block']}"
            )

            if not os.path.exists(pdb_out) or os.path.getsize(pdb_out) == 0:
                raise RuntimeError("Preprocess produced empty 'pdb'. Check chain selection / filters.")

            progress.value += 1

            # Run read.py in run_dir so outputs land there
            cmd = ["python3", READ_PY, "pdb", "alpha.cor", "coordinates"]
            _show_log(f"Running: {' '.join(cmd)}")
            proc = subprocess.run(cmd, cwd=state["run_dir"], capture_output=True, text=True)

            if proc.stdout.strip():
                _show_log(proc.stdout.rstrip())
            if proc.stderr.strip():
                _show_log(proc.stderr.rstrip())

            if proc.returncode != 0:
                raise RuntimeError(f"read.py failed (return code {proc.returncode}).")

            progress.value += 1
            progress.bar_style = "success"

            _show_log("Done. Files in run folder:")
            for fn in ["pdb", "alpha.cor", "coordinates", "gnmcutoff", "anmcutoff", "rescale"]:
                p = os.path.join(state["run_dir"], fn)
                _show_log(f" - {fn}: {'OK' if os.path.exists(p) else 'MISSING'}")

        except Exception as e:
            progress.bar_style = "danger"
            _show_log(f"ERROR: {e}")

    def on_clear_clicked(_):
        pdb_code.value = ""
        input_mode.value = "code"
        state["upload_name"] = None
        state["upload_bytes"] = None
        file_lbl.value = "No file chosen"
        chains_select.options = []
        chains_select.value = ()
        all_chains.value = False
        progress.value = 0
        progress.max = 1
        progress.bar_style = ""
        state["pdb_path"] = None
        state["run_dir"] = None
        state["detected_chains"] = []
        with log_out:
            clear_output()

    btn_load.on_click(on_load_clicked)
    btn_run.on_click(on_run_clicked)
    btn_clear.on_click(on_clear_clicked)

    form_card = W.VBox([
        W.HTML('<div class="hp-card">'),
        W.HTML("<b>Input</b>"),
        input_mode,
        code_box,
        upload_box,
        btn_load,
        W.HTML("<hr>"),
        all_chains,
        chains_select,
        W.VBox([gnm_row, anm_row], layout=W.Layout(gap="8px")),
        rescale,
        progress,
        W.HBox([btn_run, btn_clear]),
        W.HTML("</div>"),
    ])

    output_card = W.VBox([
        W.HTML('<div class="hp-card"><b>Run Log</b></div>'),
        log_out,
    ])

    display(css, header, form_card, output_card)
