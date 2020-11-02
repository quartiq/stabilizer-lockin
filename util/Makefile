../src/atan_table.txt ../src/sin_table.txt: table_gen
	./table_gen

table_gen: table_gen.rs
	rustc table_gen.rs
