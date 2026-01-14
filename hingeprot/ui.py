# ui.py
from __future__ import annotations

import os
import re
import subprocess
import datetime
import base64
import uuid
import shutil
import glob
import zipfile

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
    READ_PY          = os.path.join(HINGEPROT_DIR, "read.py")
    GNM_PY           = os.path.join(HINGEPROT_DIR, "gnm.py")
    ANM2_PY          = os.path.join(HINGEPROT_DIR, "anm2.py")
    USEBLZ_PY        = os.path.join(HINGEPROT_DIR, "useblz.py")
    ANM3_PY          = os.path.join(HINGEPROT_DIR, "anm3.py")
    EXTRACT_PY       = os.path.join(HINGEPROT_DIR, "extract.py")
    COOR2PDB_PY      = os.path.join(HINGEPROT_DIR, "coor2pdb.py")

    # Optional python ports (if present)
    PROCESSHINGES_PY = os.path.join(HINGEPROT_DIR, "processHinges.py")
    SPLITTER_PY      = os.path.join(HINGEPROT_DIR, "splitter.py")
    HINGEAA_PY       = os.path.join(HINGEPROT_DIR, "hingeaa.py")

    # NEW: rigid parts reporter (your script)
    RIGIDPARTS_PY    = os.path.join(HINGEPROT_DIR, "rigidparts_report.py")

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

    def _detect_chains(pdb_path: str) -> list[str]:
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

    def _run(cmd: list[str], cwd: str, title: str, allow_fail: bool = False):
        _show_log(f"Running: {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        if proc.stdout and proc.stdout.strip():
            _show_log(proc.stdout.rstrip())
        if proc.stderr and proc.stderr.strip():
            _show_log(proc.stderr.rstrip())
        if proc.returncode != 0:
            msg = f"{title} failed (return code {proc.returncode})."
            if allow_fail:
                _show_log("WARNING: " + msg)
                return proc
            raise RuntimeError(msg)
        return proc

    def _require_files(run_dir: str, relpaths: list[str], step_name: str):
        missing = [rp for rp in relpaths if not os.path.exists(os.path.join(run_dir, rp))]
        if missing:
            raise RuntimeError(f"{step_name}: missing required files: {', '.join(missing)}")

    def _preprocess_pdb(
        pdb_in: str,
        pdb_out: str,
        chains_keep: set[str] | None,
        keep_ter: bool = True,
        keep_only_first_model: bool = True,
    ) -> dict:
        """
        - Keep ONLY first MODEL block (if present)
        - Keep only ATOM (and optionally TER). Drop HETATM and everything else.
        - altLoc: keep if blank or 'A'
        - iCode : keep if blank
        - chain : keep only selected chains (if chains_keep is not None)
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

                    if model_found and not in_model:
                        continue

                if rec == "HETATM":
                    stats["hetatm_skipped"] += 1
                    continue

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

                if rec != "ATOM  ":
                    stats["nonatom_skipped"] += 1
                    continue

                if chains_keep is not None:
                    if len(line) <= 21:
                        stats["chain_skipped"] += 1
                        continue
                    ch = line[21]
                    if ch not in chains_keep:
                        stats["chain_skipped"] += 1
                        continue

                altloc = line[16] if len(line) > 16 else " "
                if altloc not in (" ", "A"):
                    stats["altloc_skipped"] += 1
                    continue

                icode = line[26] if len(line) > 26 else " "
                if icode != " ":
                    stats["icode_skipped"] += 1
                    continue

                fout.write(line if line.endswith("\n") else line + "\n")
                stats["lines_written"] += 1
                stats["atom_written"] += 1

        return stats

    def _list_or_custom_float(
        label: str,
        options,
        default_value: float,
        minv: float,
        maxv: float,
        step: float = 0.1,
        label_width: str = "120px",
        toggle_width: str = "180px",
        value_width: str = "240px",
    ):
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
            style={"button_width": "80px"},
        )
        dropdown = W.Dropdown(options=opts, value=default_value, layout=W.Layout(width=value_width))
        fbox = W.BoundedFloatText(
            value=default_value, min=minv, max=maxv, step=step, layout=W.Layout(width=value_width)
        )
        value_box = W.Box([dropdown], layout=W.Layout(align_items="center"))

        def _on_toggle(ch):
            value_box.children = [dropdown] if ch["new"] == "list" else [fbox]

        toggle.observe(_on_toggle, names="value")

        def get_value() -> float:
            return float(dropdown.value) if toggle.value == "list" else float(fbox.value)

        row = W.HBox([lbl, toggle, value_box], layout=W.Layout(align_items="center", gap="12px"))
        return row, get_value

    def _read_summary_text(run_dir: str, tag: str) -> str:
        """
        Prefer TAG.rigidparts.txt (full report).
        Fallback to TAG.hinge / hinges if report not produced.
        """
        candidates = [
            os.path.join(run_dir, f"{tag}.rigidparts.txt"),
            os.path.join(run_dir, f"{tag}.hinge"),
            os.path.join(run_dir, "hinges"),
        ]
        fp = next((p for p in candidates if os.path.exists(p) and os.path.getsize(p) > 0), None)
        if not fp:
            return "Summary file not found (expected: TAG.rigidparts.txt or TAG.hinge)."

        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()

        # keep it safe for UI: cap very long files
        if len(lines) > 600:
            head = lines[:300]
            tail = lines[-300:]
            lines = head + ["", "[... truncated ...]", ""] + tail

        return "\n".join(lines)

    def _make_gnm_crosscor_zip(run_dir: str, tag: str) -> str | None:
        rels = []
        for i in range(1, 11):
            a = f"crosscorrslow{i}"
            b = f"{tag}.crossslow{i}"
            if os.path.exists(os.path.join(run_dir, a)) and os.path.getsize(os.path.join(run_dir, a)) > 0:
                rels.append(a)
            elif os.path.exists(os.path.join(run_dir, b)) and os.path.getsize(os.path.join(run_dir, b)) > 0:
                rels.append(b)

        if not rels:
            return None

        zip_name = "GNM_CROSSCOR.zip"
        zp = os.path.join(run_dir, zip_name)
        try:
            if os.path.exists(zp):
                os.remove(zp)
            with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
                for rf in rels:
                    z.write(os.path.join(run_dir, rf), arcname=os.path.basename(rf))
            return zip_name
        except Exception as e:
            _show_log(f"WARNING: failed to create {zip_name}: {e}")
            return None

    def _colab_download(abs_path: str):
        from google.colab import files  # colab-only
        files.download(abs_path)

    # ---------- UI ----------
    css = W.HTML(r"""
    <style>
    .hp-card {border:1px solid #e5e7eb; border-radius:14px; padding:14px 16px; margin:10px 0; background:#fff;}
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
    .hp-dot{ width:14px; height:14px; background:#ef4444; border-radius:999px; }
    .hp-title{
      font-size:34px; font-weight:900; letter-spacing:0.5px; line-height:1.0; margin:0;
      color:#111827; font-family: Arial, Helvetica, sans-serif;
    }
    .hp-title .prot{ color:#ef4444; }
    .hp-underline{ height:3px; width:280px; background:#111827; margin-top:6px; border-radius:999px; opacity:0.9; }
    .hp-tagline{ margin-top:6px; font-size:16px; font-weight:800; color:#dc2626; font-family: Arial, Helvetica, sans-serif; }
    .hp-pre{ white-space:pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
             font-size: 13px; line-height: 1.35; background:#0b1020; color:#e5e7eb; padding:12px; border-radius:12px; border:1px solid #1f2937;}
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

    input_mode = W.ToggleButtons(
        options=[("Enter PDB code", "code"), ("Upload PDB file", "upload")],
        value="code",
        description="Input:",
        style={"description_width": "60px", "button_width": "170px"},
        layout=W.Layout(width="420px"),
    )

    pdb_code = W.Text(
        value="",
        description="PDB code:",
        placeholder="e.g., 4cln",
        style={"description_width": "80px"},
        layout=W.Layout(width="420px"),
    )

    btn_choose_file = W.Button(description="Choose file", icon="upload", layout=W.Layout(width="180px"))
    file_lbl = W.Label("No file chosen")

    code_box = W.HBox([pdb_code], layout=W.Layout(align_items="center"))
    upload_box = W.HBox([btn_choose_file, file_lbl], layout=W.Layout(align_items="center", gap="10px"))

    btn_load = W.Button(
        description="Load / Detect Chains",
        button_style="info",
        icon="search",
        layout=W.Layout(width="260px"),
    )

    all_chains = W.Checkbox(
        value=False,
        description="All Chains",
        indent=False,
        style={"description_width": "initial"},
        layout=W.Layout(width="120px", min_width="120px", flex="0 0 120px"),
    )

    chains_label = W.HTML("<b>Select Chains:</b>", layout=W.Layout(width="120px"))
    chains_wrap = W.Box(
        [],
        layout=W.Layout(
            flex="1 1 auto",
            width="auto",
            min_width="220px",
            display="flex",
            flex_flow="row wrap",
            align_items="center",
            gap="10px",
            border="1px solid #e5e7eb",
            border_radius="12px",
            padding="8px 10px",
        ),
    )

    chain_row = W.HBox([chains_label, chains_wrap], layout=W.Layout(align_items="center", gap="12px", width="100%"))

    gnm_row, get_gnm_cut = _list_or_custom_float(
        "GNM cutoff (Å):", options=[7, 8, 9, 10, 11, 12, 13, 20], default_value=10.0, minv=1.0, maxv=100.0
    )
    anm_row, get_anm_cut = _list_or_custom_float(
        "ANM cutoff (Å):", options=[10, 13, 15, 18, 20, 23, 36], default_value=18.0, minv=1.0, maxv=100.0
    )

    progress = W.IntProgress(value=0, min=0, max=1, description="Progress:", bar_style="")
    btn_run = W.Button(description="Run", button_style="success", icon="play", layout=W.Layout(width="320px"))
    btn_clear = W.Button(description="Clear", button_style="warning", icon="trash", layout=W.Layout(width="180px"))

    # Internal log (NOT displayed)
    log_out = W.Output()

    # What you want to see
    hinge_box = W.HTML('<div class="hp-pre">Report will appear here after run.</div>')
    downloads_wrap = W.VBox([], layout=W.Layout(gap="8px"))

    def _set_hinge_text(text: str):
        safe = (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        hinge_box.value = f'<div class="hp-pre">{safe}</div>'

    state = {
        "pdb_path": None,
        "run_dir": None,
        "upload_name": None,
        "upload_bytes": None,
        "detected_chains": [],
        "chain_cbs": {},
        "manual_selection": (),
        "_syncing": False,
        "pdb_tag": None,
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

    # uploader callback (unique)
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

    # ---------- chain selection logic ----------
    def _selected_chains() -> list[str]:
        detected = state.get("detected_chains", [])
        return [ch for ch in detected if ch in state["chain_cbs"] and state["chain_cbs"][ch].value]

    def _set_selection(sel: list[str]):
        detected = state.get("detected_chains", [])
        sel = [c for c in sel if c in detected]

        state["_syncing"] = True
        try:
            for ch, cb in state["chain_cbs"].items():
                cb.value = (ch in sel)
        finally:
            state["_syncing"] = False

    def _update_all_checkbox_from_selection():
        if state["_syncing"]:
            return
        detected = state.get("detected_chains", [])
        if not detected:
            return

        sel = _selected_chains()
        all_now = (len(sel) == len(detected)) and (len(detected) > 0)

        state["_syncing"] = True
        try:
            all_chains.value = all_now
        finally:
            state["_syncing"] = False

        if not all_now:
            state["manual_selection"] = tuple(sel)

    def _on_chain_cb_change(_):
        _update_all_checkbox_from_selection()

    def _rebuild_chain_checkboxes(chains: list[str], default_selected: list[str]):
        state["chain_cbs"] = {}
        items = [all_chains]
        for ch in chains:
            cb = W.Checkbox(
                value=(ch in default_selected),
                description=ch,
                indent=False,
                layout=W.Layout(width="48px", flex="0 0 48px"),
            )
            cb.observe(_on_chain_cb_change, names="value")
            state["chain_cbs"][ch] = cb
            items.append(cb)
        chains_wrap.children = items

    def _on_all_chains_toggle(ch):
        if state["_syncing"]:
            return

        detected = state.get("detected_chains", [])
        if not detected:
            return

        if ch["new"] is True:
            sel = _selected_chains()
            if len(sel) != len(detected):
                state["manual_selection"] = tuple(sel)
            _set_selection(detected)
        else:
            prev = list(state.get("manual_selection") or [])
            prev = [c for c in prev if c in detected]
            if not prev:
                prev = [detected[0]]
            _set_selection(prev)

        _update_all_checkbox_from_selection()

    all_chains.observe(_on_all_chains_toggle, names="value")

    # ---------- actions ----------
    def on_load_clicked(_):
        _set_hinge_text("Report will appear here after run.")
        downloads_wrap.children = ()

        try:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(runs_root, f"run_{ts}")
            os.makedirs(run_dir, exist_ok=True)
            state["run_dir"] = run_dir

            pdb_path = os.path.join(run_dir, "input.pdb")

            if input_mode.value == "upload":
                if state["upload_bytes"] is None:
                    raise ValueError("Please click 'Choose file' and upload a PDB first.")
                with open(pdb_path, "wb") as f:
                    f.write(state["upload_bytes"])
            else:
                code = pdb_code.value.strip()
                if not code:
                    raise ValueError("Please enter a PDB code (e.g., 4cln).")
                _fetch_pdb(code, pdb_path)

            state["pdb_path"] = pdb_path

            if input_mode.value == "code":
                tag = pdb_code.value.strip().upper()
            else:
                base = os.path.splitext(os.path.basename(state.get("upload_name") or "UPLOAD"))[0]
                tag = re.sub(r"[^0-9A-Za-z]+", "", base).upper() or "UPLOAD"
            state["pdb_tag"] = tag

            chs = _detect_chains(pdb_path)
            if not chs:
                raise RuntimeError("No chains detected in the PDB.")
            state["detected_chains"] = chs

            default_sel = [chs[0]]
            state["manual_selection"] = tuple(default_sel)
            _rebuild_chain_checkboxes(chs, default_sel)

            state["_syncing"] = True
            try:
                all_chains.value = False
            finally:
                state["_syncing"] = False

        except Exception as e:
            _set_hinge_text(f"ERROR: {e}")

    def on_run_clicked(_):
        _set_hinge_text("Running... (report will appear after completion)")
        downloads_wrap.children = ()

        try:
            if not state["pdb_path"] or not os.path.exists(state["pdb_path"]):
                raise RuntimeError("Please click 'Load / Detect Chains' first.")

            for p, name in [
                (READ_PY, "read.py"),
                (GNM_PY, "gnm.py"),
                (ANM2_PY, "anm2.py"),
                (USEBLZ_PY, "useblz.py"),
                (ANM3_PY, "anm3.py"),
                (EXTRACT_PY, "extract.py"),
                (COOR2PDB_PY, "coor2pdb.py"),
            ]:
                if not os.path.exists(p):
                    raise RuntimeError(f"{name} not found at: {p}")

            detected = state.get("detected_chains", [])
            if not detected:
                raise RuntimeError("No detected chains. Load again.")

            if all_chains.value:
                chain_list = detected
            else:
                chain_list = _selected_chains()
                if not chain_list:
                    raise RuntimeError("Please select at least one chain (or tick All chains).")

            chain_str = "".join(chain_list)
            chains_keep = None if all_chains.value else set(chain_str)

            gnm_val = float(get_gnm_cut())
            anm_val = float(get_anm_cut())

            RESCALE_DEFAULT = 1.0

            progress.max = 11
            progress.value = 0
            progress.bar_style = "info"

            run_dir = state["run_dir"]
            tag = state.get("pdb_tag") or "RUN"

            def _safe_rename(src: str, dst: str) -> bool:
                sp = os.path.join(run_dir, src)
                dp = os.path.join(run_dir, dst)
                if not os.path.exists(sp):
                    return False
                try:
                    if os.path.exists(dp):
                        os.remove(dp)
                    shutil.move(sp, dp)
                    return True
                except Exception:
                    return False

            def _zip_make(zip_name: str, rel_files: list[str]):
                zp = os.path.join(run_dir, zip_name)
                files_ok = []
                for rf in rel_files:
                    fp = os.path.join(run_dir, rf)
                    if os.path.exists(fp) and os.path.getsize(fp) > 0:
                        files_ok.append(rf)

                if not files_ok:
                    return

                if os.path.exists(zp):
                    os.remove(zp)
                with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
                    for rf in files_ok:
                        z.write(os.path.join(run_dir, rf), arcname=rf)

            # 1) params
            _write_text(os.path.join(run_dir, "gnmcutoff"), gnm_val)
            _write_text(os.path.join(run_dir, "anmcutoff"), anm_val)
            _write_text(os.path.join(run_dir, "rescale"), RESCALE_DEFAULT)
            progress.value += 1

            # 2) preprocess -> pdb
            pdb_out = os.path.join(run_dir, "pdb")
            _preprocess_pdb(
                pdb_in=state["pdb_path"],
                pdb_out=pdb_out,
                chains_keep=chains_keep,
                keep_ter=True,
                keep_only_first_model=True,
            )
            progress.value += 1

            if not os.path.exists(pdb_out) or os.path.getsize(pdb_out) == 0:
                raise RuntimeError("Preprocess produced empty 'pdb'. Check chain selection / filters.")

            # 3) read.py
            _run(["python3", READ_PY, "pdb", "alpha.cor", "coordinates"], cwd=run_dir, title="read.py")
            progress.value += 1

            # 4) gnm.py
            _run(
                ["python3", GNM_PY, "--coords", "coordinates", "--cutoff", "gnmcutoff", "--nslow", "10"],
                cwd=run_dir,
                title="gnm.py",
            )
            progress.value += 1

            # 5) anm2.py
            _run(
                ["python3", ANM2_PY, "--alpha", "alpha.cor", "--cutoff", "anmcutoff", "--out", "upperhessian"],
                cwd=run_dir,
                title="anm2.py",
            )
            progress.value += 1

            upper_path = os.path.join(run_dir, "upperhessian")
            if not os.path.exists(upper_path) or os.path.getsize(upper_path) == 0:
                raise RuntimeError("upperhessian is missing or empty; cannot solve eigenproblem.")

            # 6) useblz.py
            _run(["python3", USEBLZ_PY, "upperhessian", "--out", "upperhessian.vwmatrixd"], cwd=run_dir, title="useblz.py")
            progress.value += 1

            out_vw = os.path.join(run_dir, "upperhessian.vwmatrixd")
            if not os.path.exists(out_vw) or os.path.getsize(out_vw) == 0:
                raise RuntimeError("useblz.py did not produce upperhessian.vwmatrixd (missing/empty).")

            # 7) anm3.py
            _run(["python3", ANM3_PY, "--alpha", "alpha.cor", "--eig", "upperhessian.vwmatrixd", "--outdir", "."], cwd=run_dir, title="anm3.py")
            progress.value += 1

            _require_files(run_dir, ["newcoordinat.mds"], "anm3.py postcheck")
            _require_files(run_dir, [f"{k}coor" for k in range(1, 11)], "anm3.py postcheck")

            # 8) extract.py
            _require_files(run_dir, ["coordinates", "alpha.cor", "slowmodes"], "extract.py precheck")
            _require_files(run_dir, ["newcoordinat.mds"], "extract.py precheck")
            _run(["python3", EXTRACT_PY], cwd=run_dir, title="extract.py")
            progress.value += 1

            hinges_path = os.path.join(run_dir, "hinges")
            if not os.path.exists(hinges_path) or os.path.getsize(hinges_path) == 0:
                raise RuntimeError("extract.py did not produce hinges (missing/empty).")

            # 9) coor2pdb.py
            _require_files(run_dir, ["pdb"], "coor2pdb.py precheck")
            _require_files(run_dir, ["gnm1anmvector", "gnm2anmvector"], "coor2pdb.py precheck")
            _require_files(run_dir, [f"{k}coor" for k in range(1, 37)], "coor2pdb.py precheck")
            _run(["python3", COOR2PDB_PY], cwd=run_dir, title="coor2pdb.py")
            progress.value += 1

            # 10) rename + zips
            rename_map = [
                ("gnm1anmvector",  f"{tag}.1vector"),
                ("gnm2anmvector",  f"{tag}.2vector"),
                ("hinges",         f"{tag}.hinge"),
                ("newcoordinat.mds", f"{tag}.new"),
                ("sortedeigen",      f"{tag}.eigengnm"),
                ("eigenanm",         f"{tag}.eigenanm"),
            ]
            for k in range(1, 37):
                rename_map.append((f"{k}coor", f"{tag}.anm{k}vector"))

            for src, dst in rename_map:
                _safe_rename(src, dst)

            # Zip ANM pdb outputs (if any)
            anm_pdbs = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(run_dir, "*anm.pdb")))]
            if anm_pdbs:
                _zip_make("anm136.zip", anm_pdbs)

            gnm_cross_zip = _make_gnm_crosscor_zip(run_dir, tag)
            progress.value += 1

            # 10.5) NEW: build rigid parts report (hinges + short flexible fragments)
            # This fixes your missing text problem.
            if os.path.exists(RIGIDPARTS_PY):
                # create TAG.rigidparts.txt
                _run(
                    ["python3", RIGIDPARTS_PY, tag,
                     "--hinge", f"{tag}.hinge",
                     "--new", f"{tag}.new",
                     "--min-len", "15",
                     "--out", f"{tag}.rigidparts.txt"],
                    cwd=run_dir,
                    title="rigidparts_report.py",
                    allow_fail=True
                )
            else:
                _show_log("WARNING: rigidparts_report.py not found; will not print rigid parts/short fragments.")

            # 11) processHinges (optional) + outputs you want
            loop_thr = "15"
            clust_thr = "14.0"

            def _moved1_candidates():
                return [
                    f"{tag}.new.moved1.pdb", f"{tag}.moved1.pdb",
                    "new.moved1.pdb", "moved1.pdb",
                ]

            def _moved2_candidates():
                return [
                    f"{tag}.new.moved2.pdb", f"{tag}.moved2.pdb",
                    "new.moved2.pdb", "moved2.pdb",
                ]

            if os.path.exists(PROCESSHINGES_PY):
                def _run_processhinges(v1: str, v2: str):
                    _run(
                        ["python3", PROCESSHINGES_PY, f"{tag}.new", f"{tag}.hinge", v1, v2, loop_thr, clust_thr],
                        cwd=run_dir,
                        title="processHinges.py",
                        allow_fail=True
                    )

                # ANM pairs -> anment*.pdb
                for i in range(1, 37, 2):
                    v1 = f"{tag}.anm{i}vector"
                    v2 = f"{tag}.anm{i+1}vector"
                    if not (os.path.exists(os.path.join(run_dir, v1)) and os.path.exists(os.path.join(run_dir, v2))):
                        continue
                    _run_processhinges(v1, v2)
                    # rename if produced
                    c1 = next((c for c in _moved1_candidates() if os.path.exists(os.path.join(run_dir, c))), None)
                    c2 = next((c for c in _moved2_candidates() if os.path.exists(os.path.join(run_dir, c))), None)
                    if c1:
                        _safe_rename(c1, f"anment{i}.pdb")
                    if c2:
                        _safe_rename(c2, f"anment{i+1}.pdb")

                # GNM vectors -> moved PDBs -> mode1.ent/mode2.ent
                if os.path.exists(os.path.join(run_dir, f"{tag}.1vector")) and os.path.exists(os.path.join(run_dir, f"{tag}.2vector")):
                    _run_processhinges(f"{tag}.1vector", f"{tag}.2vector")

                moved1_now = next((c for c in _moved1_candidates() if os.path.exists(os.path.join(run_dir, c))), None)
                moved2_now = next((c for c in _moved2_candidates() if os.path.exists(os.path.join(run_dir, c))), None)
                if moved1_now:
                    shutil.copyfile(os.path.join(run_dir, moved1_now), os.path.join(run_dir, "mode1.ent"))
                if moved2_now:
                    shutil.copyfile(os.path.join(run_dir, moved2_now), os.path.join(run_dir, "mode2.ent"))

                # panm136.zip
                anment_files = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(run_dir, "anment*.pdb")))]
                if anment_files:
                    _zip_make("panm136.zip", anment_files)

            progress.value = progress.max
            progress.bar_style = "success"

            # ---------- OUTPUT: show report (rigid parts + hinges + short fragments) ----------
            report_txt = _read_summary_text(run_dir, tag)
            _set_hinge_text(report_txt)

            # ---------- DOWNLOAD buttons ----------
            buttons = []

            def _add_dl(label: str, relpath: str):
                abs_path = os.path.join(run_dir, relpath)
                if not (os.path.exists(abs_path) and os.path.getsize(abs_path) > 0):
                    return
                b = W.Button(description=label, icon="download", button_style="success", layout=W.Layout(width="320px"))
                def _cb(_):
                    _colab_download(abs_path)
                b.on_click(_cb)
                buttons.append(b)

            _add_dl("Download ANM (panm136.zip)", "panm136.zip")
            _add_dl("Download mode1.ent", "mode1.ent")
            _add_dl("Download mode2.ent", "mode2.ent")
            if gnm_cross_zip:
                _add_dl("Download GNM_CROSSCOR.zip", gnm_cross_zip)

            if not buttons:
                buttons = [W.HTML("<i>No downloadable outputs found yet.</i>")]

            downloads_wrap.children = tuple(buttons)

        except Exception as e:
            progress.bar_style = "danger"
            _set_hinge_text(f"ERROR: {e}\n\n(If you want debug logs back, tell me and I’ll show log_out.)")
            downloads_wrap.children = ()

    def on_clear_clicked(_):
        pdb_code.value = ""
        input_mode.value = "code"
        state["upload_name"] = None
        state["upload_bytes"] = None
        file_lbl.value = "No file chosen"

        state["detected_chains"] = []
        state["chain_cbs"] = {}
        state["manual_selection"] = ()
        state["_syncing"] = False
        state["pdb_tag"] = None

        all_chains.value = False
        chains_wrap.children = ()

        progress.value = 0
        progress.max = 1
        progress.bar_style = ""

        state["pdb_path"] = None
        state["run_dir"] = None

        _set_hinge_text("Report will appear here after run.")
        downloads_wrap.children = ()

    btn_load.on_click(on_load_clicked)
    btn_run.on_click(on_run_clicked)
    btn_clear.on_click(on_clear_clicked)

    form_card = W.VBox([
        W.HTML('<div class="hp-card">'),
        input_mode,
        code_box,
        upload_box,
        btn_load,
        W.HTML("<hr>"),
        chain_row,
        W.VBox([gnm_row, anm_row], layout=W.Layout(gap="8px")),
        progress,
        W.HBox([btn_run, btn_clear]),
        W.HTML("</div>"),
    ])

    output_card = W.VBox([
        W.HTML('<div class="hp-card"><b>Rigid Parts / Hinges / Short Flexible Fragments</b></div>'),
        hinge_box,
        W.HTML('<div class="hp-card"><b>Downloads</b></div>'),
        downloads_wrap,
    ])

    display(css, header, form_card, output_card)
