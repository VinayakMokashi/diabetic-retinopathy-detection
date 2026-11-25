"""Diabetic retinopathy grading from colour fundus photographs."""

__version__ = "1.0.0"

# International Clinical Diabetic Retinopathy (ICDR) severity scale, as used by EyePACS.
GRADES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}
NUM_CLASSES = len(GRADES)

# Grade 2 or worse is "referable": the patient should be seen by an eye specialist.
REFERABLE_GRADE = 2
