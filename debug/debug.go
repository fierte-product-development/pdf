// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Reading of PDF tokens and objects from a raw byte stream.

package main

import (
	"fmt"

	"github.com/rsc.io/pdf"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

func main() {
	reader, _ := pdf.Open("./test.pdf")
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
		for _, l := range contents.Body.Line {
			xy := l.ToXY()
			min := plotter.XY{X: xy[0].X, Y: xy[0].Y}
			max := plotter.XY{X: xy[1].X, Y: xy[1].Y}
			plotutil.AddLinePoints(plt, "", plotter.XYs{min, max})
		}
		w := vg.Length(plt.X.Max/100) * vg.Inch
		h := vg.Length(plt.Y.Max/100) * vg.Inch
		fName := fmt.Sprintf("page%v.png", i)
		plt.Save(w, h, fName)
	}
}
