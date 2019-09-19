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
	"sync"
	"time"

	"github.com/rsc.io/pdf"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

func parsePdf(fileName string, index int, log bool) []pdf.Contents {
	reader, _ := pdf.Open(fmt.Sprintf("./%v.pdf", fileName))
	var doc []pdf.Contents
	for i := 1; i <= reader.NumPage(); i++ {
		pg := reader.Page(i)
		cont := pg.Contents()
		mb := pg.V.Key("MediaBox")
		if log {
			saveLinePng(&mb, &cont, fileName, i)
		}
		doc = append(doc, cont)
	}
	return doc
}

func saveLinePng(mb *pdf.Value, cont *pdf.Contents, name string, index int) {
	plt, _ := plot.New()
	plt.Add(plotter.NewGrid())
	plt.X.Min = mb.Index(0).Float64()
	plt.Y.Min = mb.Index(1).Float64()
	plt.X.Max = mb.Index(2).Float64()
	plt.Y.Max = mb.Index(3).Float64()
	parts := [3]*pdf.Content{
		&cont.Header,
		&cont.Footer,
		&cont.Body,
	}
	for _, p := range parts {
		for _, t := range p.Table {
			for _, c := range t.Cell {
				draw(plt, [4]float64{c.Min.X, c.Max.X, c.Max.Y, c.Max.Y}, 0, 0)
				draw(plt, [4]float64{c.Max.X, c.Max.X, c.Min.Y, c.Max.Y}, 0, 0)
				draw(plt, [4]float64{c.Min.X, c.Max.X, c.Min.Y, c.Min.Y}, 0, 0)
				draw(plt, [4]float64{c.Min.X, c.Min.X, c.Min.Y, c.Max.Y}, 0, 0)
			}
		}
		for _, l := range p.Line {
			xy := l.ToXY()
			draw(plt, [4]float64{xy[0].X, xy[1].X, xy[0].Y, xy[1].Y}, 2, 2)
		}
	}
	w := vg.Length(plt.X.Max/100) * vg.Inch
	h := vg.Length(plt.Y.Max/100) * vg.Inch
	pngName := fmt.Sprintf("%v_page_%v.png", name, index)
	plt.Save(w, h, pngName)
}

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
	log := false
	var fileNames []string
	for i := 1; i <= 10; i++ {
		fileNames = append(fileNames, fmt.Sprintf("test_%v", i))
	}
	sTime := time.Now()
	var wg sync.WaitGroup
	docs := make([][]pdf.Contents, len(fileNames))
	for i, fn := range fileNames {
		wg.Add(1)
		go func(fileName string, index int) {
			docs[index] = parsePdf(fileName, index, log)
			wg.Done()
		}(fn, i)
	}
	wg.Wait()
	js, _ := json.MarshalIndent(docs, "", "  ")
	eTime := time.Now()
	if log {
		ioutil.WriteFile("docs.json", js, os.ModePerm)
	}
	fmt.Printf("%f秒\n", (eTime.Sub(sTime)).Seconds())
}
