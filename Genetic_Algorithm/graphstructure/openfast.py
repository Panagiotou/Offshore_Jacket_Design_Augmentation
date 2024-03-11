from pathlib import Path
import subprocess

from .graph3d import SubstructureInterface
from .opensees.structural_test import MultiNodeTest # bad style, TODO make own test for openfast

def substructure_to_openfast(sub: SubstructureInterface, test: MultiNodeTest):
    """
        Creates input files for openFAST's subdyn driver:
            - driver file
                - ENVIRONMENTAL CONDITIONS
                - SubDyn
                - INPUTS
                - STEADY INPUTS (for InputsMod = 1)
                - LOADS
            - structure file
                - SIMULATION CONTROL
                - FEA and CRAIG-BAMPTON PARAMETERS
                - STRUCTURE JOINTS: joints connect structure members (~Hydrodyn Input File)
                - BASE REACTION JOINTS: 1/0 for Locked/Free DOF @ each Reaction Node
                - INTERFACE JOINTS: 1/0 for Locked (to the TP)/Free DOF @each Interface Joint (only Locked-to-TP implemented thus far (=rigid TP))
                - MEMBERS
                - MEMBER X-SECTION PROPERTY data 1/2 [isotropic material for now: use this table for circular-tubular elements]
                - MEMBER X-SECTION PROPERTY data 2/2 [isotropic material for now: use this table if any section other than circular, however provide COSM(i,j) below]
                - CABLE PROPERTIES
                - RIGID LINK PROPERTIES
                - MEMBER COSINE MATRICES COSM(i,j)
                - JOINT ADDITIONAL CONCENTRATED MASSES
                - OUTPUT: SUMMARY & OUTFILE
                - MEMBER OUTPUT LIST
                - SDOutList: The next line(s) contains a list of output parameters that will be output in <rootname>.SD.out or <rootname>.out.
            - unsteady load file (optional)

        Returns driver filename that the command can be run on.
    """
    pass


def write_driver_file(name_base="test"):
    """
        Creates an openFAST driver file.

        Returns the filename and the OutRootName.
    """

    GRAVITY = 9.81
    WATER_DEPTH = 60
    input_file_name = f"{name_base}.dat"
    out_root_name = f"openFAST/{name_base}"

    with open(input_file_name, "w") as f:
        f.write("Compatible with SubDyn v1.00.00\n")
        f.write("FALSE\tEcho\n") # dont print the file's information to the console
        f.write(f"{GRAVITY}\tGravity\n")
        f.write(f"{WATER_DEPTH}\tWtrDpth\n")
        f.write(f'"{name_base}.drv"\tSDInputFile\n')
        f.write(f'"{out_root_name}"\tOutRootName\n')

        f.write(f"600\tNSteps\n")
        f.write(f"0.001\tTimeInterval\n")
        f.write(f"0.0\t0.0\t18.15\tTP_RefPoint\n")
        f.write(f"0.0\tSubRotateZ\n")
        f.write(f"1\tInputsMod\n")
        f.write(f'""\tInputsFile\n')

        # TODO: What are these?
        f.write(f"3.821E-02\t1.656E-03\t-4.325E-02\t-1.339E-04\t7.266E-02\t-2.411E-03\tuTPInSteady\n")
        f.write(f"1.02\t2.03\t5.03\t0.03\t0.03\t0.03\tuDotTPInSteady\n")
        f.write(f"2.02\t3.03\t-9.03\t0.3\t0.03\t0.3\tuDotDotTPInSteady\n")
        f.write(f"1\tnAppliedLoads\n")

        # TODO automate from test loads
        f.write(f"ALJointID\tFx\tFy\tFz\tMx\tMy\tMz\tUnsteadyFile\n")
        f.write(f"(-)\t(N)\t(N)\t(N)\t(Nm)\t(Nm)\t(Nm)\t(-)\n")
        f.write(f'13\t100\t0\t0\t0\t0\t0\t""\n')
        f.write("END of driver input file\n")

    return out_root_name


def execute_openfast(drive_file: Path):
    """
        Takes driver and input files for openFAST and runs the subdyn driver on them.
    """
    subprocess.run(["subdyn_driver", drive_file])
