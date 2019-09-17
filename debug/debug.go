// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Reading of PDF tokens and objects from a raw byte stream.

package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/rsc.io/pdf"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

func draw(plt *plot.Plot, points [4]float64, color int, dashes int) {
	min := plotter.XY{X: points[0], Y: points[2]}
	max := plotter.XY{X: points[1], Y: points[3]}
	l, s, _ := plotter.NewLinePoints(plotter.XYs{min, max})
	l.Color = plotutil.Color(color)
	l.Dashes = plotutil.Dashes(dashes)
	s.Color = plotutil.Color(color)
	plt.Add(l, s)
}

func main() {
	reader, _ := pdf.Open("./test.pdf")
	var doc []pdf.Contents
	var docs [][]pdf.Contents
	for i := 1; i <= reader.NumPage(); i++ {
		plt, _ := plot.New()
		plt.Add(plotter.NewGrid())
		pg := reader.Page(i)
		mb := pg.V.Key("MediaBox")
		plt.X.Min = mb.Index(0).Float64()
		plt.Y.Min = mb.Index(1).Float64()
		plt.X.Max = mb.Index(2).Float64()
		plt.Y.Max = mb.Index(3).Float64()
		contents := pg.Contents()
		for _, t := range contents.Body.Table {
			for _, c := range t.Cell {
				draw(plt, [4]float64{c.Min.X, c.Max.X, c.Max.Y, c.Max.Y}, 0, 0)
				draw(plt, [4]float64{c.Max.X, c.Max.X, c.Min.Y, c.Max.Y}, 0, 0)
				draw(plt, [4]float64{c.Min.X, c.Max.X, c.Min.Y, c.Min.Y}, 0, 0)
				draw(plt, [4]float64{c.Min.X, c.Min.X, c.Min.Y, c.Max.Y}, 0, 0)
			}
		}
		for _, l := range contents.Body.Line {
			xy := l.ToXY()
			draw(plt, [4]float64{xy[0].X, xy[1].X, xy[0].Y, xy[1].Y}, 2, 2)
		}
		w := vg.Length(plt.X.Max/100) * vg.Inch
		h := vg.Length(plt.Y.Max/100) * vg.Inch
		fName := fmt.Sprintf("page%v.png", i)
		plt.Save(w, h, fName)
		doc = append(doc, contents)
	}
	docs = append(docs, doc)
	js, _ := json.MarshalIndent(docs, "", "  ")
	ioutil.WriteFile("docs.json", js, os.ModePerm)
}
