import xlrd#读取excel


event = xlrd.open_workbook('C:\consumer_portrait\excel\EVENT.xlsx')

event_data = event.sheets()[0]          #通过索引顺序获取

event_data.row_values(1)#第一行
event_data.col_values(1)#第一列





import xlwt#写入excel
event_data_result = xlwt.Workbook(encoding = 'ascii')
event_data_sheet = event_data_result.add_sheet('event')
event_data_sheet .write(0, 0, label = 1)

event_data_result.save('C:\consumer_portrait\\result\\event_data_result.xls')