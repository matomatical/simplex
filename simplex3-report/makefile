report.html: report.md $(wildcard %.png) distill.html makefile
	pandoc --to html --template distill.html --filter pandoc-katex --from markdown -o $@ $<
