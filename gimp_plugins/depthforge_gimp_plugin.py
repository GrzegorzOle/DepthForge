#!/usr/bin/env python3
import sys

with open("/tmp/depthforge-loaded.log", "a", encoding="utf-8") as f:
    f.write("plugin loaded\n")

import gi
gi.require_version("Gimp", "3.0")
from gi.repository import Gimp, GLib

PROC_NAME = "python-fu-depthforge"


class DepthForgePlugin(Gimp.PlugIn):
    def do_query_procedures(self):
        return [PROC_NAME]

    def do_set_i18n(self, procname):
        return False, None, None

    def do_create_procedure(self, name):
        procedure = Gimp.ImageProcedure.new(
            self,
            name,
            Gimp.PDBProcType.PLUGIN,
            self.run,
            None
        )
        procedure.set_image_types("*")
        procedure.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.DRAWABLE)
        procedure.set_menu_label("Depth Map Generator")
        procedure.add_menu_path("<Image>/Filters/DepthForge/")
        procedure.set_documentation(
            "Generate depth maps from images",
            "Generate depth maps from images for tactile visualization",
            name
        )
        procedure.set_attribution(
            "DepthForge Team",
            "DepthForge Team",
            "2026"
        )
        return procedure

    def run(self, procedure, run_mode, image, drawables, config, run_data):
        with open("/tmp/depthforge-run.log", "a", encoding="utf-8") as f:
            f.write("run called\n")

        return procedure.new_return_values(
            Gimp.PDBStatusType.SUCCESS,
            GLib.Error()
        )


Gimp.main(DepthForgePlugin.__gtype__, sys.argv)