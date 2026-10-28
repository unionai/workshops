def view_file(file_path, title=None, show_path=False, base_dir=None, code_style="one-dark"):
    from pathlib import Path
    from pygments import highlight
    from pygments.lexers import PythonLexer
    from pygments.formatters import HtmlFormatter
    from IPython.display import HTML, display

    p = Path(file_path)
    code = p.read_text(encoding="utf-8")

    formatter = HtmlFormatter(style=code_style, noclasses=True)
    html_code = highlight(code, PythonLexer(), formatter)

    if show_path:
        base = Path(base_dir) if base_dir else Path.cwd()
        try:
            display_path = p.resolve().relative_to(base.resolve())
        except Exception:
            display_path = p.name

        left_label = f'<span style="opacity:.7; margin-right:10px;">{title}</span>' if title else ""

        header_html = f"""
        <div style="
          display:flex; align-items:center; justify-content:space-between;
          margin: 8px 0 6px 0;
          font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
          font-size: 12px; opacity:.95;
        ">
          <div style="display:flex; align-items:center;">
            {left_label}
            <span style="
              padding: 4px 10px;
              border-radius: 999px;
              border: 1px solid rgba(127,127,127,0.25);
              background: rgba(127,127,127,0.10);
            ">{display_path}</span>
          </div>
          <div></div>
        </div>
        """
        display(HTML(header_html))

    display(HTML(html_code))
