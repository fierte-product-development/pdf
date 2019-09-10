// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Reading of PDF tokens and objects from a raw byte stream.

package main

import (
	"github.com/rsc.io/pdf"
	/*
		"fmt"
		"github.com/gonum/plot"
		"github.com/gonum/plot/plotter"
		"github.com/gonum/plot/plotutil"
		"github.com/gonum/plot/vg"
	*/)

func main() {
	reader, _ := pdf.Open("./test.pdf")
	for i := 1; i <= reader.NumPage(); i++ {
		/*
			plt, _ := plot.New()
			plt.Add(plotter.NewGrid())
		*/
		pg := reader.Page(i)
		/*
			mb := pg.V.Key("MediaBox")
			plt.X.Min = mb.Index(0).Float64()
			plt.Y.Min = mb.Index(1).Float64()
			plt.X.Max = mb.Index(2).Float64()
			plt.Y.Max = mb.Index(3).Float64()
		*/
		pg.Contents()
		/*
			for _, l := range contents.Body.Line {
				min := plotter.XY{X: l.Min.X, Y: l.Min.Y}
				max := plotter.XY{X: l.Max.X, Y: l.Max.Y}
				plotutil.AddLinePoints(plt, "", plotter.XYs{min, max})
			}
			w := vg.Length(plt.X.Max/100) * vg.Inch
			h := vg.Length(plt.Y.Max/100) * vg.Inch
			fName := fmt.Sprintf("page%v.png", i)
			plt.Save(w, h, fName)
		*/
	}
}
