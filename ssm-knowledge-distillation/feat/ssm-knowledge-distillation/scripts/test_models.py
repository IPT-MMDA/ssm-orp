import torch
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from teacher import TeacherSSM
from student import StudentSSM
from distill import kd_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_teacher_forward():
    model = TeacherSSM(n_classes=35).to(DEVICE)
    x = torch.randn(2, 16000, device=DEVICE)
    out = model(x)
    assert out.shape == (2, 35), "Teacher output shape mismatch: " + str(out.shape)
    print("PASS: teacher forward")


def test_student_forward():
    model = StudentSSM(n_classes=35).to(DEVICE)
    x = torch.randn(2, 16000, device=DEVICE)
    out = model(x)
    assert out.shape == (2, 35), "Student output shape mismatch: " + str(out.shape)
    print("PASS: student forward")


def test_student_smaller_than_teacher():
    teacher = TeacherSSM(n_classes=35)
    student = StudentSSM(n_classes=35)
    t_params = sum(p.numel() for p in teacher.parameters())
    s_params = sum(p.numel() for p in student.parameters())
    assert s_params < t_params, "Student should have fewer params than teacher"
    print("PASS: student (", s_params, ") < teacher (", t_params, ")")


def test_kd_loss_alpha_zero():
    #alpha=0 should give pure cross entropy
    student_logits = torch.randn(4, 10)
    teacher_logits = torch.randn(4, 10)
    labels = torch.randint(0, 10, (4,))
    loss_kd = kd_loss(student_logits, teacher_logits, labels, temperature=1, alpha=0.0)
    loss_ce = F.cross_entropy(student_logits, labels)
    assert torch.allclose(loss_kd, loss_ce, atol=1e-5), "alpha=0 should equal CE"
    print("PASS: kd_loss alpha=0 equals CE")


def test_kd_loss_temperature():
    #higher temperature should give different loss
    student_logits = torch.randn(4, 10)
    teacher_logits = torch.randn(4, 10)
    labels = torch.randint(0, 10, (4,))
    loss_t1 = kd_loss(student_logits, teacher_logits, labels, temperature=1, alpha=1.0)
    loss_t10 = kd_loss(student_logits, teacher_logits, labels, temperature=10, alpha=1.0)
    assert not torch.allclose(loss_t1, loss_t10), "Different temperatures should give different losses"
    print("PASS: kd_loss varies with temperature")


def test_kd_loss_gradient():
    #loss should be differentiable
    student_logits = torch.randn(4, 10, requires_grad=True)
    teacher_logits = torch.randn(4, 10)
    labels = torch.randint(0, 10, (4,))
    loss = kd_loss(student_logits, teacher_logits, labels, temperature=2, alpha=0.5)
    loss.backward()
    assert student_logits.grad is not None, "Should have gradients"
    print("PASS: kd_loss is differentiable")


if __name__ == "__main__":
    test_teacher_forward()
    test_student_forward()
    test_student_smaller_than_teacher()
    test_kd_loss_alpha_zero()
    test_kd_loss_temperature()
    test_kd_loss_gradient()
    print()
    print("All tests passed!")
