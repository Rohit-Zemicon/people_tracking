import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GObject, GLib
import cairo

Gst.init(None)


def draw_callback(overlay, context, timestamp, duration):
    # Set line style for polygon
    context.set_source_rgb(1.0, 1.0, 0.0)  # Yellow
    context.set_line_width(4.0)

    # Polygon points
    points = [(3, 530), (640, 530), (600, 200), (30, 200)]

    # Draw polygon
    context.move_to(points[0][0], points[0][1])
    for x, y in points[1:]:
        context.line_to(x, y)
    context.close_path()
    context.stroke()

    # Draw text label
    context.set_source_rgb(1.0, 0.0, 0.0)  # Red text
    context.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    context.set_font_size(32)

    text = "Restricted Area"
    x, y = points[2]  # Place text near top point
    context.move_to(x + 10, y - 10)
    context.show_text(text)


pipeline = Gst.parse_launch(
    "rtspsrc location=rtsp://127.0.0.1:8554/mystream latency=0 ! "
    "decodebin ! videoconvert ! cairooverlay name=overlay ! autovideosink"
)

overlay = pipeline.get_by_name("overlay")
overlay.connect("draw", draw_callback)

pipeline.set_state(Gst.State.PLAYING)
print("Piprline started....")

loop = GLib.MainLoop()
try:
    loop.run()
except KeyboardInterrupt:
    pass

pipeline.set_state(Gst.State.NULL)
