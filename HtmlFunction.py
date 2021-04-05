from typing import List


def htmlf(textParams: List[tuple]) -> str:
    profiles = {
        1: """<span class="s1">{}</span>""",
        2: """<span class="s2">{}</span>""",
        3: """<span class="s3">{}</span>""",
        4: """<span class="s4">{}</span>""",
        "b": """<span class="b">{}</span>""",
        "i": """<span class="i">{}</span>""",
        "ib": """<span class="ib">{}</span>"""
    }

    out = []
    for textParam in textParams:
        text = textParam[0].replace("\n", "<br/>")
        profile = textParam[1]
        htmlString = profiles[profile].format(text)
        out.append(htmlString)

    outStr = "".join(out)

    return f"""
    <html>
        <head>
            <style>
                span.i {{
                    font-size:40px;
                    font-style:italic;
                }}
                span.b {{
                    font-size:40px;
                    font-weight:bold;
                }}
                span.ib {{
                    font-size:40px;
                    font-style:oblique;
                    font-weight:bold;
                }}
                span.s1 {{
                    font-size:10px;
                }}
                span.s2 {{
                    font-size:20px;
                }}
                span.s3 {{
                    font-size:30px;
                }}
                span.s4 {{
                    font-size:40px;
                }}
            </style>
        </head>
        <body>
            {outStr}
        </body>
    </html>"""


if __name__ == "__main__":
    print(htmlf([("Hello\n", 1), ("Bye", 3)]))
