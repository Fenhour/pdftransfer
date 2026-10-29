"""Legacy local appearance rendering, used only during project import."""
import html

def esc(value):
    return html.escape(str(value), quote=True)


def edited_text(element):
    return (element.get("render_mode") == "text" or
            element.get("text", "") != element.get("original_text", "") or
            element.get("style", {}) != element.get("original_style", element.get("style", {})))


def local_content(element):
    width, height = element["base_width"], element["base_height"]
    style = element.get("style", {})
    if element["type"] == "text" and edited_text(element):
        size = style.get("font_size", 12)
        anchor = {"left": "start", "center": "middle", "right": "end"}.get(style.get("align"), "start")
        x = {"start": 0, "middle": width/2, "end": width}[anchor]
        rotation = element.get("source", {}).get("text_rotation", 0)
        # Original quad/appearance remains available; regenerated text follows its baseline.
        baseline = element.get("text_origin", [0, size * .84])
        if anchor == "start":
            x = baseline[0]
        y = baseline[1]
        runs = []
        for index, line in enumerate(element.get("text", "").split("\n")):
            runs.append(f'<tspan x="{x}" y="{y + index * size * style.get("line_height", 1.2)}">{esc(line)}</tspan>')
        return (f'<text fill="{esc(style.get("color", "#222222"))}" font-family="{esc(style.get("font_family", "sans-serif"))}" '
                f'font-size="{size}" font-weight="{esc(style.get("font_weight", "normal"))}" '
                f'font-style="{esc(style.get("font_style", "normal"))}" text-anchor="{anchor}" '
                f'transform="rotate({rotation},{x},{y})" xml:space="preserve">{"".join(runs)}</text>')
    appearance = element.get("appearance", {})
    if appearance.get("kind") == "image":
        return f'<image width="{width}" height="{height}" preserveAspectRatio="none" href="{esc(appearance["data"])}"/>'
    if appearance.get("kind") == "svg":
        return appearance.get("svg", "")
    if element["type"] == "line" and element.get("source", {}).get("orientation") == "vertical":
        return f'<line x1="{width/2}" y1="0" x2="{width/2}" y2="{height}" stroke="{esc(style.get("stroke", "#202020"))}" stroke-width="{style.get("stroke_width", 1)}"/>'
    if element["type"] == "line":
        return f'<line x1="0" y1="{height/2}" x2="{width}" y2="{height/2}" stroke="{esc(style.get("stroke", "#202020"))}" stroke-width="{style.get("stroke_width", 1)}"/>'
    return f'<rect width="{width}" height="{height}" fill="{esc(style.get("fill", "#dce9e8"))}" stroke="{esc(style.get("stroke", "none"))}" stroke-width="{style.get("stroke_width", 1)}"/>'

