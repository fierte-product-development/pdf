// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Reading of PDF tokens and objects from a raw byte stream.

package main

import (
	"fmt"

	"github.com/rsc.io/pdf"
)

type size struct {
	x float64
	y float64
}

func main() {
	reader, err := pdf.Open("./test.pdf")
	for i := 1; i <= reader.NumPage(); i++ {
		page := reader.Page(i)
		mb := page.V.Key("MediaBox")
		pageSize := size{mb.Index(2).Float64(), mb.Index(3).Float64()}
		contents := page.Content()
		fmt.Printf("%v\n", contents)
		fmt.Printf("%v\n", pageSize)
	}
	fmt.Printf("%v\n", err)
}
