# -*- coding: utf-8 -*-
from openerp import models, fields, api#, exceptions
import base64,xlrd,os #,StringIO,io,openpyxl
#import xlrd,tkFileDialog,Tkinter
#from io import StringIO
#from openpyxl import workbook
#from openpyxl import load_workbook

class PunchTask(models.Model):
    _name = 'punch.task'
    _description = 'Punch task'
    datafile=fields.Binary(u'Excel(.xlsx)',required=True)
    records=fields.Text('有误记录')
    #rawData=xlrd.open_workbook('/home/caofa/test.xls')
    #table=rawData.sheets()[0]
    #x=table.row_values(2)

    @api.multi
    def select_odd(self,cr):
        filename='/tmp/%s.xlsx' % cr['uid']
        data_file_p=open(filename,'w')                # this method can get data by transportation of a part of disk
        #data_file_p.write((base64.b64decode(self.datafile)))
        data_file_p.write((base64.decodestring(self.datafile)))
        data_file_p.close()
        wb=xlrd.open_workbook(filename)
        os.remove(filename)
        table=wb.sheets()[0]
        col_name=table.col_values(1)[1:]
        col_datetime=table.col_values(3)[1:]
        col_date=[]
        col_time=[]
        for c in col_datetime:
            col_date.append(c[:10])
            col_time.append(c[11:])
        date_unique=list(set(col_date))
        name_unique=list(set(col_name))
        duty_on=[]
        duty_off=[]
        for i in range(len(col_time)):
            if col_time[i]<'08:31:00':
                duty_on.append(i)
            elif col_time[i]>'17:00:00':
                duty_off.append(i)

        duty_on_record=[]
        for i in range(len(name_unique)):
            duty_on_record.append([])
        
        duty_on_record_copy=duty_on_record

        for i in duty_on:
            for index in range(len(name_unique)):
                if col_name[i]==name_unique[index]:
                    duty_on_record[index].append(col_date[i])
                    break

        duty_on_output=[]
        for i in range(len(duty_on_record)):
           for date in date_unique:
               if date not in duty_on_record[i]:
                   #duty_on_output.append(','.join([name_unique[i],date]))
                   duty_on_record_copy

        self.records='\n'.join(duty_on_output)

'''
        #data=StringIO.StringIO(base64.b64decode(self.datafile)) # thie method can get data by transportation of computer memories and can only read xlsx file rather that xlsx file;
        #data=io.BytesIO(base64.b64decode(self.datafile))
        data=StringIO.StringIO(base64.decodestring(self.datafile))
        wb=load_workbook(data)
        table=wb.worksheets[0]
        tem=table.rows[2]
        self.records=tem[2]
'''
'''
        data_file_p=open('/tmp/test.xlsx','w')                # this method can get data by transportation of a part of disk
        #data_file_p.write((base64.b64decode(self.datafile)))
        data_file_p.write((base64.decodestring(self.datafile)))
        data_file_p.close()
        wb=xlrd.open_workbook(filename=r'/tmp/test.xlsx')
        table=wb.sheets()[0]
        tem=table.row_values(2)
        self.records=tem[2]
'''


    
'''
    def select_odd(self,cr,uid,ids,context=None):
        #try:
        #data=StringIO(self.browse(cr,uid,ids)[0].datafile.decode('utf-8'))
        #wb=load_workbook(data)
        #except:
        #    raise exceptions.Warning(u'错误！'+u'您输入的不是xlsx文件！')
        #ws=wb.worksheets[0]
'''

'''
    #@api.onchange("datafile")
    def select_odd(self,cr,uid,ids,context=None):
        this=self.browse(cr,uid,ids[0])
        #data = base64.decodestring(this.datafile)
        data = this.datafile.decode('base64')
        this.records=data

        with open(data,mode='rb') as f:
            filedata=f.read()
            tem=filedata
            this.records=tem
        #raise exceptions.Warning(self.x)
'''
'''
    @api.multi
    def select_odd(self):
        #root=Tkinter.Tk()
        #dirname=tkFileDialog.askdirectory(parent=root,initialdir="/",title='Please select a directory')
        #if len(dirname)>0:
        #    print "you choose %s" % dirname
        master=Tkinter.Tk()# these two lines delect tk window
        master.withdraw()
      
        filename=tkFileDialog.askopenfilenames(parent=master,filetypes=[("Excel file","*.xls"),("All","*.*")],title="选择文件") # "parent" can confirm to select many times; and if use these codes, when user click the button, poped window which shows to select the target file shold be server rather than user;
        if len(filename)>0:
            rawData=xlrd.open_workbook(filename[0])
            table=rawData.sheets()[0]
            tem=table.row_values(2)
            self.records=tem
'''
    



    




