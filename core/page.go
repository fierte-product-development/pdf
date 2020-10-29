// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package core

import (
	"encoding/hex"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"strings"

	"github.com/rakyll/statik/fs"
)

// A Page represent a single page in a PDF file.
// The methods interpret a Page dictionary stored in V.
type Page struct {
	V Value
}

// Page returns the page for the given page number.
// Page numbers are indexed starting at 1, not 0.
// If the page is not found, Page returns a Page with p.V.IsNull().
func (r *Reader) Page(num int) Page {
	num-- // now 0-indexed
	page := r.Trailer().Key("Root").Key("Pages")
Search:
	for page.Key("Type").Name() == "Pages" {
		count := int(page.Key("Count").Int64())
		if count < num {
			return Page{}
		}
		kids := page.Key("Kids")
		for i := 0; i < kids.Len(); i++ {
			kid := kids.Index(i)
			if kid.Key("Type").Name() == "Pages" {
				c := int(kid.Key("Count").Int64())
				if num < c {
					page = kid
					continue Search
				}
				num -= c
				continue
			}
			if kid.Key("Type").Name() == "Page" {
				if num == 0 {
					return Page{V: kid}
				}
				num--
			}
		}
		break
	}
	return Page{}
}

// NumPage returns the number of pages in the PDF file.
func (r *Reader) NumPage() int {
	return int(r.Trailer().Key("Root").Key("Pages").Key("Count").Int64())
}

func (p *Page) findInherited(key string) Value {
	for v := p.V; !v.IsNull(); v = v.Key("Parent") {
		if r := v.Key(key); !r.IsNull() {
			return r
		}
	}
	return Value{}
}

// MediaBox returns bounding box of page.
func (p *Page) MediaBox() BoundingBox {
	return *newBoundingBox(&p.V)
}

/*
func (p *Page) CropBox() Value {
	return p.findInherited("CropBox")
}
*/

// Resources returns the resources dictionary associated with the page.
func (p *Page) Resources() Value {
	return p.findInherited("Resources")
}

// Fonts returns a list of the fonts associated with the page.
func (p *Page) Fonts() []string {
	return p.Resources().Key("Font").Keys()
}

// Font returns the font with the given name associated with the page.
func (p *Page) Font(name string) Font {
	return Font{p.Resources().Key("Font").Key(name)}
}

// A Font represent a font in a PDF file.
// The methods interpret a Font dictionary stored in V.
type Font struct {
	V Value
}

// BaseFont returns the font's name (BaseFont property).
func (f Font) BaseFont() string {
	return f.V.Key("BaseFont").Name()
}

// FirstChar returns the code point of the first character in the font.
func (f Font) FirstChar() int {
	return int(f.V.Key("FirstChar").Int64())
}

// LastChar returns the code point of the last character in the font.
func (f Font) LastChar() int {
	return int(f.V.Key("LastChar").Int64())
}

// Widths returns the widths of the glyphs in the font.
// In a well-formed PDF, len(f.Widths()) == f.LastChar()+1 - f.FirstChar().
func (f Font) Widths() []float64 {
	x := f.V.Key("Widths")
	var out []float64
	for i := 0; i < x.Len(); i++ {
		out = append(out, x.Index(i).Float64())
	}
	return out
}

// Width returns the width of the given code point.
func (f Font) Width(code int) float64 {
	first := f.FirstChar()
	last := f.LastChar()
	if code < first || last < code {
		return 0
	}
	return f.V.Key("Widths").Index(code - first).Float64()
}

// Encoder returns the encoding between font code point sequences and UTF-8.
func (f Font) Encoder() [2]TextEncoding {
	enc := f.V.Key("Encoding")
	te := [2]TextEncoding{}
	switch enc.Kind() {
	case Name:
		switch enc.Name() {
		case "WinAnsiEncoding":
			te[0] = &byteEncoder{&winAnsiEncoding}
		case "MacRomanEncoding":
			te[0] = &byteEncoder{&macRomanEncoding}
		case "UniJIS-UTF16-V", "UniJIS-UTF16-H", "UniJIS-UCS2-V", "UniJIS-UCS2-H":
			te[0] = &utf16beEncoder{}
			te[1] = getCmap(enc.Name())
		default:
			toUnicode := f.V.Key("ToUnicode")
			switch toUnicode.Kind() {
			case Stream, Dict:
				cm := readCmap(toUnicode.Reader())
				if cm == nil {
					println("nil cmap", enc.Name())
					te[0] = &nopEncoder{}
				}
				te[0] = cm
			default:
				println("unknown encoding", enc.Name())
				te[0] = &nopEncoder{}
			}
		}
	case Dict:
		te[0] = &dictEncoder{enc.Key("Differences")}
	case Null:
		// ok, try ToUnicode
	default:
		println("unexpected encoding", enc.String())
		te[0] = &nopEncoder{}
	}

	return te
	// return &byteEncoder{&pdfDocEncoding}
}

type dictEncoder struct {
	v Value
}

func (e *dictEncoder) Decode(raw string) (text string) {
	r := make([]rune, 0, len(raw))
	for i := 0; i < len(raw); i++ {
		ch := rune(raw[i])
		n := -1
		for j := 0; j < e.v.Len(); j++ {
			x := e.v.Index(j)
			if x.Kind() == Integer {
				n = int(x.Int64())
				continue
			}
			if x.Kind() == Name {
				if int(raw[i]) == n {
					r := nameToRune[x.Name()]
					if r != 0 {
						ch = r
						break
					}
				}
				n++
			}
		}
		r = append(r, ch)
	}
	return string(r)
}

// A TextEncoding represents a mapping between
// font code points and UTF-8 text.
type TextEncoding interface {
	// Decode returns the UTF-8 text corresponding to
	// the sequence of code points in raw.
	Decode(raw string) (text string)
}

type nopEncoder struct {
}

func (e *nopEncoder) Decode(raw string) (text string) {
	return raw
}

type byteEncoder struct {
	table *[256]rune
}

func (e *byteEncoder) Decode(raw string) (text string) {
	r := make([]rune, 0, len(raw))
	for i := 0; i < len(raw); i++ {
		r = append(r, e.table[raw[i]])
	}
	return string(r)
}

type cmap struct {
	space   [4][][2]string
	bfrange []bfrange
}

func (m *cmap) append(srcLo string, srcHi string, dst string) {
	b, err := hex.DecodeString(dst)
	if err == nil {
		if len(dst) == 4 {
			dst = string(b)
		}
	}
	m.bfrange = append(m.bfrange, bfrange{srcLo, srcHi, dst})
}

func (m *cmap) Decode(raw string) (text string) {
	var r []rune
Parse:
	for len(raw) > 0 {
		for n := 1; n <= 4 && n <= len(raw); n++ {
			for _, space := range m.space[n-1] {
				if space[0] <= raw[:n] && raw[:n] <= space[1] {
					text := raw[:n]
					raw = raw[n:]
					for _, bf := range m.bfrange {
						if len(bf.lo) == n && bf.lo <= text && text <= bf.hi {
							s := bf.dst
							if bf.lo != text {
								b := []byte(s)
								b[len(b)-1] += text[len(text)-1] - bf.lo[len(bf.lo)-1]
								s = string(b)
							}
							r = append(r, []rune(utf16Decode(s))...)
							continue Parse
						}
					}
					fmt.Fprintf(os.Stderr, "no text for %q\n", text)
					r = append(r, noRune)
					continue Parse
				}
			}
		}
		println("no code space found")
		r = append(r, noRune)
		raw = raw[1:]
	}
	return string(r)
}

type bfrange struct {
	lo  string
	hi  string
	dst string
}

func readCmap(toUnicode io.ReadCloser) *cmap {
	n := -1
	var m cmap
	ok := true
	uni := func(val Value) string {
		switch val.Kind() {
		case String:
			return val.RawString()
		case Integer:
			return fmt.Sprintf("%U", val.Int64())[2:]
		case Array:
			fmt.Fprintf(os.Stderr, "array %v\n", val)
			return ""
		default:
			fmt.Fprintf(os.Stderr, "unknown cmap %v\n", val)
			return ""
		}
	}
	Interpret(toUnicode, &Stack{}, func(stk *Stack, op string) {
		if !ok {
			return
		}
		switch op {
		case "findresource":
			category := stk.Pop()
			key := stk.Pop()
			fmt.Sprintln("findresource", key, category)
			stk.Push(newDict())
		case "begincmap":
			stk.Push(newDict())
		case "endcmap":
			stk.Pop()
		case "begincodespacerange":
			n = int(stk.Pop().Int64())
		case "endcodespacerange":
			if n < 0 {
				println("missing begincodespacerange")
				ok = false
				return
			}
			for i := 0; i < n; i++ {
				hi, lo := stk.Pop().RawString(), stk.Pop().RawString()
				if len(lo) == 0 || len(lo) != len(hi) {
					println("bad codespace range")
					ok = false
					return
				}
				m.space[len(lo)-1] = append(m.space[len(lo)-1], [2]string{lo, hi})
			}
			n = -1
		case "beginbfrange", "begincidrange":
			n = int(stk.Pop().Int64())
		case "endbfrange", "endcidrange":
			if n < 0 {
				panic("missing beginbfrange")
			}
			for i := 0; i < n; i++ {
				dst, srcHi, srcLo := uni(stk.Pop()), uni(stk.Pop()), uni(stk.Pop())
				m.append(srcLo, srcHi, dst)
			}
		case "beginbfchar", "begincidchar":
			n = int(stk.Pop().Int64())
		case "endbfchar", "endcidchar":
			if n < 0 {
				panic("missing beginbfchar")
			}
			for i := 0; i < n; i++ {
				dst, srcHi := uni(stk.Pop()), uni(stk.Pop())
				srcLo := srcHi
				m.append(srcLo, srcHi, dst)
			}
		case "beginnotdefrange":
			n = int(stk.Pop().Int64())
		case "endnotdefrange":
			if n < 0 {
				panic("missing endnotdefrange")
			}
			for i := 0; i < n; i++ {
				dst, srcHi, srcLo := uni(stk.Pop()), uni(stk.Pop()), uni(stk.Pop())
				lo := []rune(srcLo)
				hi := []rune(srcHi)
				if lo[1] == 0 && hi[1] == 31 {
					for j := 0; j <= 31; j++ {
						r := []int32{0, int32(j)}
						m.append(string(r), string(r), dst)
					}
				} else {
					panic("Unexpected range og notdefrange")
				}
			}
		case "defineresource":
			category := stk.Pop().Name()
			value := stk.Pop()
			key := stk.Pop().Name()
			fmt.Sprintln("defineresource", key, value, category)
			stk.Push(value)
		default:
			println("interp\t", op)
		}
	})
	if !ok {
		return nil
	}
	return &m
}

func getCmap(name string) *cmap {
	statikFS, _ := fs.New()
	r, _ := statikFS.Open("/" + name + ".txt")
	defer r.Close()
	return readCmap(r)
}

type matrix [3][3]float64

var ident = matrix{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}

func (x matrix) mul(y matrix) matrix {
	var z matrix
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			for k := 0; k < 3; k++ {
				z[i][j] += x[i][k] * y[k][j]
			}
		}
	}
	return z
}

// A Text is List of Char.
type Text struct {
	Char []*Char
	BoundingBox
}

func (t *Text) append(c *Char) {
	if len(t.Char) == 0 {
		t.Min = c.Min
		t.Max = c.Max
	} else {
		if t.Min.X > c.Min.X {
			t.Min.X = c.Min.X
		}
		if t.Min.Y > c.Min.Y {
			t.Min.Y = c.Min.Y
		}
		if t.Max.X < c.Max.X {
			t.Max.X = c.Max.X
		}
		if t.Max.Y < c.Max.Y {
			t.Max.Y = c.Max.Y
		}
	}
	t.Char = append(t.Char, c)
}

// A Char represents a single piece of text drawn on a page.
type Char struct {
	Font     string  // the font used
	FontSize float64 // the font size, in points (1/72 of an inch)
	S        string  // the actual UTF-8 text
	BoundingBox
}

// A Point represents an X, Y pair.
type Point struct {
	X float64
	Y float64
}

// A BoundingBox is range where objects can be drawn.
type BoundingBox struct {
	Min Point
	Max Point
}

func newBoundingBox(v *Value) *BoundingBox {
	newbb := func(v Value) *BoundingBox {
		bbox := new(BoundingBox)
		bbox.Min.X = v.Index(0).Float64()
		bbox.Min.Y = v.Index(1).Float64()
		bbox.Max.X = v.Index(2).Float64()
		bbox.Max.Y = v.Index(3).Float64()
		return bbox
	}
	bbox := newbb(v.Key("MediaBox"))
	if bbox.IsEmpty() {
		bbox = newbb(v.Key("BBox"))
	}
	return bbox
}

// IsEmpty returns whether all coordinates are zero.
func (bbox *BoundingBox) IsEmpty() bool {
	return bbox.Min.X == 0 &&
		bbox.Min.Y == 0 &&
		bbox.Max.X == 0 &&
		bbox.Max.Y == 0
}

// Contains returns whether or not all points passed are within range of the bbox.
func (bbox *BoundingBox) Contains(pts ...Point) bool {
	for _, pt := range pts {
		if bbox.Min.X-1.0 > pt.X ||
			bbox.Min.Y-1.0 > pt.Y ||
			bbox.Max.X+1.0 < pt.X ||
			bbox.Max.Y+1.0 < pt.Y {
			return false
		} //数値は調整
	}
	return true
}

// Points is used when you want to pass a bbox to Contains.
func (bbox *BoundingBox) Points() (Point, Point) {
	return bbox.Min, bbox.Max
}

// A fontInfos is Font information that you want to get only once.
type fontInfos struct {
	Name    string
	Encoder [2]TextEncoding
	Width   map[int]float64
	DWidth  float64 // Default
	Bytes   int
}

// CreateText creates a Text object from a string(Tj or TJ argument).
func (fi *fontInfos) CreateText(s string, g *gstate) *Text {
	gid := fi.getGid(s)
	text := Text{}
	CORR := 1000. // グリフ幅の1はテキスト空間の1/1000のサイズを表すため

	n := 0
	for _, ch := range fi.Encoder[0].Decode(s) {
		Trm := matrix{{g.Tfs * g.Th, 0, 0}, {0, g.Tfs, 0}, {0, g.Trise, 1}}.mul(g.Tm).mul(g.CTM)
		w0 := fi.Width[gid[n]]
		if w0 == 0 {
			w0 = fi.DWidth
		}
		isSpace := strings.TrimSpace(string(ch)) == ""
		if !isSpace {
			char := Char{
				fi.Name, Trm[0][0], string(ch),
				BoundingBox{
					Point{Trm[2][0], Trm[2][1]},
					Point{Trm[2][0] + (w0 / CORR * Trm[0][0]), Trm[2][1] + Trm[1][1]},
				},
			}
			text.append(&char)
		}
		tx := w0/CORR*g.Tfs + g.Tc
		if isSpace {
			tx += g.Tw
		}
		tx *= g.Th
		g.Tm = matrix{{1, 0, 0}, {0, 1, 0}, {tx, 0, 1}}.mul(g.Tm)
		n++
	}
	return &text
}

func (fi *fontInfos) getGid(s string) []int {
	// TODO: TextEncoderのDecodeから取得したい
	chrs := []int{}
	if fi.Encoder[1] != nil {
		s = fi.Encoder[1].Decode(s)
		for _, r := range s {
			chrs = append(chrs, int(r))
		}
		return chrs
	}
	bytes := []byte(s)
	buf := 0
	for i, r := range bytes {
		mod := (i + 1) % fi.Bytes
		if mod == 0 {
			chrs = append(chrs, buf+int(r))
			buf = 0
		} else {
			buf2 := int(r)
			for c := 0; c < (fi.Bytes - mod); c++ {
				buf2 *= 256
			}
			buf += buf2
		}
	}
	return chrs
}

// A fontInfo interface is used in combination with fontInfos.
type fontInfo interface {
	setWidth(*Font)
	getFontInfos() fontInfos
}

// A type1Font is a ASCII font.
type type1Font struct {
	fontInfos
}

// A type0Font is mainly used for Japanese fonts.
type type0Font struct {
	fontInfos
}

// newFontInfo is the constructor of FontInfo. Processing branches depending on the value of "Subtype".
func newFontInfo(f *Font) fontInfo {
	var fi fontInfo
	switch st := f.V.Key("Subtype").Name(); st {
	case "TrueType", "Type1":
		fi = newType1Font(f)
	case "Type0":
		fi = newType0Font(f)
	default:
		println("Unknown Font")
	}
	return fi
}

// newType1Font is called by newFontInfo.
func newType1Font(f *Font) *type1Font {
	fi := new(type1Font)
	fi.Name = f.BaseFont()
	fi.Encoder = f.Encoder()
	fi.Bytes = 1
	fi.setWidth(f)
	return fi
}

func (tp1 *type1Font) setWidth(f *Font) {
	widthsMap := map[int]float64{}
	first := f.FirstChar()
	last := f.LastChar()
	widths := f.V.Key("Widths")
	for i := 0; i <= (last - first); i++ {
		widthsMap[first+i] = widths.Index(i).Float64()
	}
	tp1.Width = widthsMap
	tp1.DWidth = f.V.Key("FontDescriptor").Key("MissingWidth").Float64() // default = 0
}

func (tp1 *type1Font) getFontInfos() fontInfos {
	return tp1.fontInfos
}

// newType0Font is called by newFontInfo.
func newType0Font(f *Font) *type0Font {
	fi := new(type0Font)
	fi.Name = f.BaseFont()
	fi.Encoder = f.Encoder()
	fi.Bytes = 2
	fi.setWidth(f)
	return fi
}

func (tp0 *type0Font) setWidth(f *Font) {
	widthsMap := map[int]float64{}
	df := f.V.Key("DescendantFonts").Index(0)
	widths := df.Key("W")
	entryIdx := 1
	first := 0
	last := 0
	for i := 0; i < widths.Len(); i++ {
		if entryIdx == 1 {
			first = int(widths.Index(i).Int64())
			entryIdx++
		} else if entryIdx == 2 {
			if secondEntry := widths.Index(i); secondEntry.Kind() == Array {
				for j := 0; j < secondEntry.Len(); j++ {
					widthsMap[first+j] = secondEntry.Index(j).Float64()
				}
				entryIdx = 1 // all done
			} else {
				last = int(secondEntry.Int64())
				entryIdx++
			}
		} else if entryIdx == 3 {
			val := widths.Index(i).Float64()
			for c := first; c <= last; c++ {
				widthsMap[c] = val
			}
			entryIdx = 1 // all done
		}
	}
	tp0.Width = widthsMap
	if dw := df.Key("DW").Float64(); dw != 0 {
		tp0.DWidth = dw
	} else {
		tp0.DWidth = 1000
	}
}

func (tp0 *type0Font) getFontInfos() fontInfos {
	return tp0.fontInfos
}

// A Line represents a line
type Line struct {
	Min Point
	Max Point
}

// 横書き(縦長)pdfを前提とするためy座標が優先、下から上、左から右
func newLine(p1 Point, p2 Point) *Line {
	l := new(Line)
	if p1.Y < p2.Y {
		l.Min = p1
		l.Max = p2
	} else if p1.Y > p2.Y {
		l.Min = p2
		l.Max = p1
	} else if p1.X < p2.X {
		l.Min = p1
		l.Max = p2
	} else if p1.X > p2.X {
		l.Min = p2
		l.Max = p1
	} else {
		panic("Same points.")
	}
	return l
}

// Polygon is a connected series of lines that form a closed shape.
type Polygon struct {
	Line []*Line
}

// Content describes the basic content on a page: the text and any drawn lines.
type Content struct {
	Text    []*Text
	Line    []*Line
	Polygon []*Polygon
}

func (c *Content) append(cont *Content) {
	c.Text = append(c.Text, cont.Text...)
	c.Line = append(c.Line, cont.Line...)
	c.Polygon = append(c.Polygon, cont.Polygon...)
}

type gstate struct {
	cs    bool
	CS    bool
	Tc    float64
	Tw    float64
	Th    float64
	Tl    float64
	Tf    string
	Tfs   float64
	Tmode int
	Trise float64
	Tm    matrix
	Tlm   matrix
	Trm   matrix
	CTM   matrix
}

func newgstate() *gstate {
	gs := new(gstate)
	gs.CS = true
	gs.cs = true
	gs.Th = 1
	gs.CTM = ident
	return gs
}

type points struct {
	p []Point
}

func (ps *points) append(p ...Point) {
	ps.p = append(ps.p, p...)
}

func (ps *points) closePath() {
	l := len(ps.p)
	if l == 0 {
		panic("point stack is empty")
	}
	if ps.p[0] != ps.p[l-1] {
		ps.p = append(ps.p, ps.p[0])
	}
}

func (ps *points) stroke() []*Line {
	var ls []*Line
	for i := 0; i < len(ps.p)-1; i++ {
		ls = append(ls, newLine(ps.p[i], ps.p[i+1]))
	}
	return ls
}

func (ps *points) fill() *Polygon {
	pg := new(Polygon)
	pg.Line = append(pg.Line, ps.stroke()...)
	return pg
}

// Contents returns the page's content.
func (p *Page) Contents() Content {
	val := p.V.Key("Contents")
	vals := []Value{}
	switch val.Kind() {
	case Array: // 古いpdfではストリームがサイズ区切りでリストになっていることがある
		for i := 0; i < val.Len(); i++ {
			vals = append(vals, val.Index(i))
		}
	case Stream:
		vals = append(vals, val)
	default:
		println(val.Kind())
	}
	return getContentFromStream(&p.V, vals, *newgstate())
}

func getContentFromStream(parent *Value, streams []Value, g gstate) Content {
	result := Content{}
	fontInfos := map[string]fontInfo{}

	mbox := *newBoundingBox(parent)
	showText := func(s string) {
		fi := fontInfos[g.Tf].getFontInfos()
		text := fi.CreateText(s, &g)
		if !text.IsEmpty() && mbox.Contains(text.Points()) {
			result.Text = append(result.Text, text)
		}
	}

	var ps points
	closePath := func() {
		ps.closePath()
	}
	stroke := func() {
		if g.CS && mbox.Contains(ps.p...) {
			result.Line = append(result.Line, ps.stroke()...)
		}
	}
	fill := func() {
		if g.cs && mbox.Contains(ps.p...) {
			result.Polygon = append(result.Polygon, ps.fill())
		}
	}

	// ページコンテンツがサイズで分割されている可能性があるため各スタックはループの外側で用意する
	var argstack Stack
	var gstack []gstate
	for _, strm := range streams {
		Interpret(strm.Reader(), &argstack, func(stk *Stack, op string) {
			n := stk.Len()
			args := make([]Value, n)
			for i := n - 1; i >= 0; i-- {
				args[i] = stk.Pop()
			}
			switch op {
			default:
				// fmt.Println(op, args)
				return
			/*
				グラフィック描画の流れ
				1. グラフィックステート(線の太さとか色とか)を変更
				2. パスを生成
				3. 1と2を元に描画(同時に2を初期化)
			*/
			// qで現在のステートを保存しQで取り出す。qとQは必ず同数存在する
			case "q":
				gstack = append(gstack, g)
			case "Q":
				n := len(gstack) - 1
				if n < 0 {
					panic("graphic state stack is empty")
				}
				g = gstack[n]
				gstack = gstack[:n]

			case "l", "m": // mはパスの始点、lは直線のパス
				if len(args) != 2 {
					panic("bad l or m")
				}
				//printStream(strm)
				ps.append(Point{
					g.CTM[2][0] + args[0].Float64()*g.CTM[0][0],
					g.CTM[2][1] + args[1].Float64()*g.CTM[1][1],
				})
			// c,v,yはそれぞれベジェ曲線を描画するためのオペレータだが都合上直線として扱う
			case "c":
				if len(args) != 6 {
					panic("bad c")
				}
				ps.append(Point{
					g.CTM[2][0] + args[4].Float64()*g.CTM[0][0],
					g.CTM[2][1] + args[5].Float64()*g.CTM[1][1],
				})
			case "v", "y":
				if len(args) != 4 {
					panic("bad v or y")
				}
				ps.append(Point{
					g.CTM[2][0] + args[2].Float64()*g.CTM[0][0],
					g.CTM[2][1] + args[3].Float64()*g.CTM[1][1],
				})
			case "h": // パスを閉じる(最後のポイントから最初のポイントまで直線のパスを引く)
				closePath()
			case "n", "b", "b*", "B", "B*", "f", "F", "f*", "S", "s":
				switch op {
				case "b", "b*", "s":
					closePath()
				}
				switch op {
				case "b", "b*", "B", "B*", "f", "F", "f*":
					fill()
				}
				switch op {
				case "b", "b*", "B", "B*", "S", "s":
					stroke()
				}
				ps = points{}

			case "re": // 四角形のパス
				if len(args) != 4 {
					panic("bad re")
				}
				x := g.CTM[2][0] + args[0].Float64()*g.CTM[0][0]
				y := g.CTM[2][1] + args[1].Float64()*g.CTM[1][1]
				w := args[2].Float64() * g.CTM[0][0]
				h := args[3].Float64() * g.CTM[1][1]
				points := []Point{
					Point{x, y + h},
					Point{x + w, y + h},
					Point{x + w, y},
					Point{x, y},
					Point{x, y + h},
				}
				ps.append(points...)

			case "gs": // 透明度などのステートが入った辞書をページオブジェクトから取得する
				gs := parent.Key("Resources").Key("ExtGState").Key(args[0].Name())
				font := gs.Key("Font")
				if !font.IsNull() {
					fmt.Fprint(os.Stderr, "ExtGState's Font operator is not implemented.")
				}

			// 塗りつぶしおよびストロークの色設定。白以外をtrueとする
			// なおRGBは加法混色のため1 1 1が白、CMYKは減法混色のため0 0 0 0が白である。
			case "cs", "CS":
			case "sc", "g", "rg", "k":
				var sum float64 = 0
				for _, arg := range args {
					sum += arg.Float64()
				}
				switch len(args) {
				case 1:
					g.cs = sum != 1
				case 3:
					g.cs = sum != 3
				case 4:
					g.cs = sum != 0
				}
			case "SC", "G", "RG", "K":
				var sum float64 = 0
				for _, arg := range args {
					sum += arg.Float64()
				}
				switch len(args) {
				case 1:
					g.CS = sum != 1
				case 3:
					g.CS = sum != 3
				case 4:
					g.CS = sum != 0
				}

			case "Do": // XObjectに記載されたオペレータを実行
				for _, arg := range args {
					xobj := parent.Key("Resources").Key("XObject").Key(arg.String()[1:])
					xg := g
					cm := xobj.Key("Matrix")
					if !cm.IsNull() {
						var m matrix
						for i := 0; i < 6; i++ {
							m[i/2][i%2] = cm.Index(i).Float64()
						}
						m[2][2] = 1
						xg.CTM = m.mul(xg.CTM)
					}
					st := xobj.Key("Subtype")
					if st.String() == "/Form" {
						xcontent := getContentFromStream(&xobj, []Value{xobj}, xg)
						result.append(&xcontent)
					}
				}

			case "cm": // update g.CTM
				if len(args) != 6 {
					panic("bad g.Tm")
				}
				var m matrix
				abcdef := [6]float64{}
				for i := 0; i < 6; i++ {
					abcdef[i] = args[i].Float64()
					m[i/2][i%2] = abcdef[i]
				}
				if !(abcdef[1] == 0 && abcdef[2] == 0) {
					fmt.Fprintf(os.Stderr, "Coordinate system is not implemented distorted shape. %v\n", abcdef)
				}
				m[2][2] = 1
				g.CTM = m.mul(g.CTM)

			case "BT": // begin text (reset text matrix and line matrix)
				g.Tm = ident
				g.Tlm = g.Tm

			case "ET": // end text

			case "T*": // move to start of next line
				x := matrix{{1, 0, 0}, {0, 1, 0}, {0, -g.Tl, 1}}
				g.Tlm = x.mul(g.Tlm)
				g.Tm = g.Tlm

			case "Tc": // set character spacing
				if len(args) != 1 {
					panic("bad g.Tc")
				}
				g.Tc = args[0].Float64()

			case "TD": // move text position and set leading
				if len(args) != 2 {
					panic("bad Td")
				}
				g.Tl = -args[1].Float64()
				fallthrough
			case "Td": // move text position
				if len(args) != 2 {
					panic("bad Td")
				}
				tx := args[0].Float64()
				ty := args[1].Float64()
				x := matrix{{1, 0, 0}, {0, 1, 0}, {tx, ty, 1}}
				g.Tlm = x.mul(g.Tlm)
				g.Tm = g.Tlm

			case "Tf": // set text font and size
				if len(args) != 2 {
					panic("bad Tf")
				}
				g.Tf = args[0].Name()
				if _, ok := fontInfos[g.Tf]; !ok {
					fontInfos[g.Tf] = newFontInfo(&Font{parent.Key("Resources").Key("Font").Key(g.Tf)})
				}
				g.Tfs = args[1].Float64()

			case "\"": // set spacing, move to next line, and show text
				if len(args) != 3 {
					panic("bad \" operator")
				}
				g.Tw = args[0].Float64()
				g.Tc = args[1].Float64()
				args = args[2:]
				fallthrough
			case "'": // move to next line and show text
				if len(args) != 1 {
					panic("bad ' operator")
				}
				x := matrix{{1, 0, 0}, {0, 1, 0}, {0, -g.Tl, 1}}
				g.Tlm = x.mul(g.Tlm)
				g.Tm = g.Tlm
				fallthrough
			case "Tj": // show text
				if len(args) != 1 {
					panic("bad Tj operator")
				}
				showText(args[0].RawString())

			case "TJ": // show text, allowing individual glyph positioning
				v := args[0]
				for i := 0; i < v.Len(); i++ {
					x := v.Index(i)
					if x.Kind() == String {
						showText(x.RawString())
					} else {
						fi := fontInfos[g.Tf].getFontInfos()
						tx := -x.Float64() / fi.DWidth * g.Tfs * g.Th
						g.Tm = matrix{{1, 0, 0}, {0, 1, 0}, {tx, 0, 1}}.mul(g.Tm)
					}
				}

			case "TL": // set text leading
				if len(args) != 1 {
					panic("bad TL")
				}
				g.Tl = args[0].Float64()

			case "Tm": // set text matrix and line matrix
				if len(args) != 6 {
					panic("bad g.Tm")
				}
				var m matrix
				for i := 0; i < 6; i++ {
					m[i/2][i%2] = args[i].Float64()
				}
				m[2][2] = 1
				g.Tm = m
				g.Tlm = m

			case "Tr": // set text rendering mode
				if len(args) != 1 {
					panic("bad Tr")
				}
				g.Tmode = int(args[0].Int64())

			case "Ts": // set text rise
				if len(args) != 1 {
					panic("bad Ts")
				}
				g.Trise = args[0].Float64()

			case "Tw": // set word spacing
				if len(args) != 1 {
					panic("bad g.Tw")
				}
				g.Tw = args[0].Float64()

			case "Tz": // set horizontal text scaling
				if len(args) != 1 {
					panic("bad Tz")
				}
				g.Th = args[0].Float64() / 100
			}
		})
	}

	return result
}

// TextVertical implements sort.Interface for sorting
// a slice of Text values in vertical order, top to bottom,
// and then left to right within a line.
type TextVertical []Text

// func (x TextVertical) Len() int      { return len(x) }
// func (x TextVertical) Swap(i, j int) { x[i], x[j] = x[j], x[i] }
// func (x TextVertical) Less(i, j int) bool {
// 	if x[i].Y != x[j].Y {
// 		return x[i].Y > x[j].Y
// 	}
// 	return x[i].X < x[j].X
// }

// TextHorizontal implements sort.Interface for sorting
// a slice of Text values in horizontal order, left to right,
// and then top to bottom within a column.
type TextHorizontal []Text

// func (x TextHorizontal) Len() int      { return len(x) }
// func (x TextHorizontal) Swap(i, j int) { x[i], x[j] = x[j], x[i] }
// func (x TextHorizontal) Less(i, j int) bool {
// 	if x[i].X != x[j].X {
// 		return x[i].X < x[j].X
// 	}
// 	return x[i].Y > x[j].Y
// }

// An Outline is a tree describing the outline (also known as the table of contents)
// of a document.
type Outline struct {
	Title string    // title for this element
	Child []Outline // child elements
}

// Outline returns the document outline.
// The Outline returned is the root of the outline tree and typically has no Title itself.
// That is, the children of the returned root are the top-level entries in the outline.
func (r *Reader) Outline() Outline {
	return buildOutline(r.Trailer().Key("Root").Key("Outlines"))
}

func buildOutline(entry Value) Outline {
	var x Outline
	x.Title = entry.Key("Title").Text()
	for child := entry.Key("First"); child.Kind() == Dict; child = child.Key("Next") {
		x.Child = append(x.Child, buildOutline(child))
	}
	return x
}

// デバッグ用
func printStream(val Value) {
	bt, _ := ioutil.ReadAll(val.Reader())
	print(string(bt))
}

// デバッグ用
func printFont(f *Font) {
	v := f.V
	print(v.Key("BaseFont"))
	print(v.Keys())
	print(v.Key("DescendantFonts").Len())
	df := v.Key("DescendantFonts").Index(0)
	print(df.Keys())
	print(df.Key("CIDSystemInfo").Keys())
	print(df.Key("FontDescriptor").Keys())
	print(df.Key("FontDescriptor").Key("Flags"))
	print(df.Key("W"))
	print(df.Key("DW"))
}

func print(a interface{}) {
	fmt.Printf("%v\n", a)
}
