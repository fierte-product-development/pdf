package main

import (
	"encoding/json"
	"flag"
	"io/ioutil"
	"os"

	"github.com/rsc.io/pdf"
	"golang.org/x/text/width"
)

func main() {
	flag.Parse()
	cmdArgs := flag.Args()
	var js []byte
	if len(cmdArgs) != 0 {
		arg := []byte(cmdArgs[0])
		js = cleanPdfText(arg)
	} else {
		stdin, _ := ioutil.ReadAll(os.Stdin)
		js = cleanPdfText(stdin)
	}
	os.Stdout.Write(js)
}

func cleanPdfText(raw []byte) []byte {
	var rawJSON []byte
	switch raw[0] {
	case '{', '[':
		rawJSON = raw
	default:
		rawJSON, _ = ioutil.ReadFile(string(raw))
	}
	var docs [][]pdf.Content
	json.Unmarshal(rawJSON, &docs)
	for i, doc := range docs {
		for j, page := range doc {
			docs[i][j].Text = cleanTexts(page.Text)
			for k, table := range page.Table {
				for l, cell := range table.Cell {
					docs[i][j].Table[k].Cell[l].Text = cleanTexts(cell.Text)
				}
			}
		}
	}
	js, _ := json.MarshalIndent(docs, "", "  ")
	return js
}

func cleanTexts(ts []pdf.Text) []pdf.Text {
	var cleanedTexts []pdf.Text
	for _, t := range ts {
		cleanedTexts = append(cleanedTexts, t)
		for _, c := range t.Char {
			switch c.S {
			// ハイフンっぽい文字を揃える
			case "–", "―", "−", "─":
				c.S = "－"
			// 全角の点を半角に
			case "・":
				c.S = "･"
			// カンマは消す
			case ",":
				c.S = ""
			// カタカナ以外を半角にする
			default:
				c.S = width.Fold.String(c.S)
			}
		}
	}
	return cleanedTexts
}
