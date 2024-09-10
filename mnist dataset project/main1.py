import streamlit as st
from streamlit.components.v1 import html

def main():
    st.title("Drawing Board App")

    # Embed the drawing canvas using HTML and JavaScript
    html_code = """
    <style>
        #canvas {
            border: 1px solid #000;
            background-color: white;
        }
    </style>
    <canvas id="canvas" width="600" height="400"></canvas>
    <script>
        var canvas = document.getElementById("canvas");
        var ctx = canvas.getContext("2d");
        var isDrawing = false;

        canvas.addEventListener("mousedown", startDrawing);
        canvas.addEventListener("mouseup", stopDrawing);
        canvas.addEventListener("mousemove", draw);

        function startDrawing(e) {
            isDrawing = true;
            draw(e);
        }

        function stopDrawing() {
            isDrawing = false;
            ctx.beginPath();
        }

        function draw(e) {
            if (!isDrawing) return;

            ctx.lineWidth = 2;
            ctx.lineCap = "round";
            ctx.strokeStyle = "black";

            ctx.lineTo(e.clientX - canvas.getBoundingClientRect().left, e.clientY - canvas.getBoundingClientRect().top);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(e.clientX - canvas.getBoundingClientRect().left, e.clientY - canvas.getBoundingClientRect().top);
        }
    </script>
    """

    # Display the HTML code
    html(html_code)

    # Button to clear the canvas
    if st.button("Clear Canvas"):
        st.experimental_rerun()

    # Button to save the drawing
    if st.button("Save Drawing"):
        save_drawing(canvas)

def save_drawing(canvas):
    # Convert the canvas to an image
    canvas_image = canvas.toDataURL("image/png")
    st.image(canvas_image, caption="Drawing", use_column_width=True, channels="RGBA")

if __name__ == '__main__':
    main()
