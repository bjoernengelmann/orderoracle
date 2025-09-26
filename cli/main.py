from __future__ import annotations

import typer

from .commands import pt_export_docs as cmd_pt_export_docs
from .commands import pt_export_topics_qrels as cmd_pt_export_tq
from .commands import embed as cmd_embed
from .commands import eval as cmd_eval
from .commands import report as cmd_report


app = typer.Typer(no_args_is_help=True, add_completion=False)

# Register subcommands from their modules to keep this file minimal
cmd_pt_export_docs.register(app)
cmd_pt_export_tq.register(app)
cmd_embed.register(app)
cmd_eval.register(app)
cmd_report.register(app)

def main():  # pragma: no cover
    app()


