Option Explicit
Dim app, doc, i, tocCount
Set app = CreateObject("Word.Application")
app.Visible = False
app.DisplayAlerts = 0
WScript.Echo "OPEN"
Set doc = app.Documents.Open(WScript.Arguments(0), False, False)
doc.Repaginate
doc.Fields.Update
tocCount = doc.TablesOfContents.Count
For i = 1 To tocCount
  doc.TablesOfContents.Item(i).Update
Next
doc.Repaginate
For i = 1 To tocCount
  doc.TablesOfContents.Item(i).UpdatePageNumbers
Next
doc.Save
WScript.Echo "EXPORT"
doc.ExportAsFixedFormat WScript.Arguments(1), 17, False, 0, 0, 1, 1, 0, True, True, 1
WScript.Echo "PAGES=" & doc.ComputeStatistics(2)
doc.Close 0
app.Quit
