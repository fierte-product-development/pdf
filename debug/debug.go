// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// デバッグ用
// json.goとだいたい同じ

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

func parsePage(fileName string, fileIdx int, log bool) []pdf.Content {
	fileName = fmt.Sprintf("./%v.pdf", fileName)
	r, _ := pdf.Open(fileName)
	np := r.NumPage()
	doc := make([]pdf.Content, np)
	var wg sync.WaitGroup
	for i := 1; i <= np; i++ {
		wg.Add(1)
		func(i int) { //go
			pg := r.Page(i)
			cont := pg.Contents()
			if log {
				mb := pg.V.Key("MediaBox")
				saveLinePng(&cont, &mb, fileIdx, i)
			}
			doc[i-1] = cont
			wg.Done()
		}(i)
	}
	wg.Wait()
	return doc
}

func saveLinePng(cont *pdf.Content, mbox *pdf.Value, fileIdx int, pageIdx int) {
	plt, _ := plot.New()
	plt.Add(plotter.NewGrid())
	plt.X.Min = mbox.Index(0).Float64()
	plt.Y.Min = mbox.Index(1).Float64()
	plt.X.Max = mbox.Index(2).Float64()
	plt.Y.Max = mbox.Index(3).Float64()
	for _, t := range cont.Table {
		for _, c := range t.Cell {
			draw(plt, [4]float64{c.Min.X, c.Max.X, c.Max.Y, c.Max.Y}, 0, 0)
			draw(plt, [4]float64{c.Max.X, c.Max.X, c.Min.Y, c.Max.Y}, 0, 0)
			draw(plt, [4]float64{c.Min.X, c.Max.X, c.Min.Y, c.Min.Y}, 0, 0)
			draw(plt, [4]float64{c.Min.X, c.Min.X, c.Min.Y, c.Max.Y}, 0, 0)
		}
	}
	for _, l := range cont.Line {
		xy := l.ToXY()
		draw(plt, [4]float64{xy[0].X, xy[1].X, xy[0].Y, xy[1].Y}, 2, 2)
	}
	w := vg.Length(plt.X.Max/100) * vg.Inch
	h := vg.Length(plt.Y.Max/100) * vg.Inch
	pngName := fmt.Sprintf("file%v_page%v.png", fileIdx+1, pageIdx)
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
	log := true
	fileNames := []string{
		"MEC報告書",
		//"AVANT短信",
		/*
			"COSMO報告書",
			"COSMO短信",
			"ID報告書",
			"ID短信",
			"MEC報告書",
			"MEC短信",
			"Wacom報告書",
			"Wacom短信",
		*/
	}
	sTime := time.Now()
	docs := make([][]pdf.Content, len(fileNames))
	var wg sync.WaitGroup
	for i, fn := range fileNames {
		wg.Add(1)
		func(fn string, i int) { //go
			docs[i] = parsePage(fn, i, log)
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
