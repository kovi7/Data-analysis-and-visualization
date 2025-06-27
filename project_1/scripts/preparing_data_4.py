import pandas as pd

data = [
    # Winter Semester
    ["Winter Semester", "Winter", "01.10.2025", "15.02.2026"],
    ["Classes (Block I)", "Winter", "02.10.2025", "02.11.2025"],
    ["Deadline to remove course link to study program for winter semester", "Winter", "21.10.2025", "21.10.2025"],
    ["Classes (Block II)", "Winter", "03.11.2025", "07.12.2025"],
    ["Course linking period for summer semester", "Winter", "01.10.2025", "15.02.2026"],
    ["Classes (Block III)", "Winter", "08.12.2025", "21.12.2025"],
    ["Winter Break", "Winter", "22.12.2025", "06.01.2026"],
    ["Classes (Block III cont.)", "Winter", "07.01.2026", "25.01.2026"],
    ["Days off", "Winter", "01.10.2025", "01.10.2025"],
    ["Days off", "Winter", "10.11.2025", "10.11.2025"],
    ["Final deadline to withdraw from winter semester course credit", "Winter", "16.01.2026", "16.01.2026"],
    ["Winter Exam Session", "Winter", "26.01.2026", "08.02.2026"],
    ["Written certification exams (foreign languages)", "Winter", "26.01.2026", "27.01.2026"],
    ["Inter-semester Break", "Winter", "09.02.2026", "15.02.2026"],
    # Summer Semester
    ["Summer Semester", "Summer", "16.02.2026", "30.09.2026"],
    ["Classes (Block I)", "Summer", "16.02.2026", "22.03.2026"],
    ["Retake Exam Session (Winter)", "Summer", "20.02.2026", "01.03.2026"],
    ["Retake written certification exams (English B2 level)", "Summer", "21.02.2026", "21.02.2026"],
    ["Decision period for winter semester completion (specific study programs)", "Summer", "02.03.2026", "29.03.2026"],
    ["Deadline to remove course link to study program for summer semester", "Summer", "13.03.2026", "13.03.2026"],
    ["Classes (Block II)", "Summer", "23.03.2026", "03.05.2026"],
    ["Spring Break", "Summer", "02.04.2026", "07.04.2026"],
    ["Classes (Block III)", "Summer", "04.05.2026", "07.06.2026"],
    ["Days off", "Summer", "02.05.2026", "03.05.2026"],
    ["Days off", "Summer", "08.05.2026", "09.05.2026"],
    ["Days off", "Summer", "05.06.2026", "05.06.2026"],
    ["Final deadline to withdraw from summer semester course credit", "Summer", "31.05.2026", "31.05.2026"],
    ["Course linking period for the next academic year (winter semester and full year)", "Summer", "01.06.2026", "30.09.2026"],
    ["Summer Exam Session", "Summer", "08.06.2026", "28.06.2026"],
    ["Written certification exams (foreign languages)", "Summer", "08.06.2026", "09.06.2026"],

    ["Summer Break", "Vacation", "29.06.2026", "30.09.2026"],
    ["Vacation Block I", "Vacation", "06.07.2026", "07.08.2026"],
    ["Vacation Block II", "Vacation", "17.08.2026", "11.09.2026"],
    ["Retake Exam Session (Summer)", "Vacation", "31.08.2026", "13.09.2026"],
    ["Written certification exams (foreign languages)", "Vacation", "31.08.2026", "01.09.2026"],
    ["Decision period for academic year completion (specific study programs)", "Vacation", "14.09.2026", "30.09.2026"]]

df = pd.DataFrame(data, columns=["Event", "Semester", "Start Date", "End Date"])
df.to_csv('../data/academic_schedule_25_26.csv', index=False)
