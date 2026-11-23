"""Các phần của scripts/aggregate_results.py, mỗi module một việc.

- constants:       tên mô hình, loại file, cột bảng, màu.
- issues:          IssueLog, sổ ghi lỗi/cảnh báo.
- records:         đọc JSON; ResultRow là một dòng của bảng.
- gpu_hours:       GpuHoursConverter, quy đổi giờ train về T4.
- fairness_checks: các hàm check_* kiểm tra so sánh công bằng.
- table_values:    giá trị từng ô của bảng chất lượng và chi phí.
- charts:          biểu đồ KID.
- figure_grids:    ghép ảnh mẫu, interpolation, nearest-neighbor.
- report:          bảng markdown, CSV, summary.md.
"""
