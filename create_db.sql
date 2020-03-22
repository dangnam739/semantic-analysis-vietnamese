CREATE TABLE labeled_paragraph(
    para_id SERIAL PRIMARY KEY,
    content TEXT NOT NULL ,
    label INTEGER NOT NULL
);