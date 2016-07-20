# -*- coding: utf-8 -*-
from openerp import models, fields, api#, exceptions
import base64,io,pandas #,StringIO,openpyxl,xlrd,os,
#import xlrd,tkFileDialog,Tkinter
#from io import StringIO
#from openpyxl import workbook
#from openpyxl import load_workbook

class PunchTask(models.Model):
    _name = 'punch.task'
    _description = 'Punch task'
    datafile=fields.Binary(u'选择考勤Excel表',required=True)
    duty_on=fields.Text(u'8:30前没打卡')
    duty_off=fields.Text(u'17:00后没打卡')
    #rawData=xlrd.open_workbook('/home/caofa/test.xls')
    #table=rawData.sheets()[0]
    #x=table.row_values(2)

    @api.multi
    def select_odd(self,cr):

        #filename='/tmp/%s.xlsx' % cr['uid']
        #data_file_p=open(filename,'w')  # this method can get data by transportation of a part of disk
        ##data_file_p.write((base64.b64decode(self.datafile)))
        #data_file_p.write((base64.decodestring(self.datafile)))
        #data_file_p.close()
        #wb=xlrd.open_workbook(filename)
        #os.remove(filename)
        #table=wb.sheets()[0]
        #col_name=table.col_values(1)[1:]
        #col_datetime=table.col_values(3)[1:]

        file_like=io.BytesIO(base64.b64decode(self.datafile))
        #datax=pandas.read_excel(file_like)
        table=pandas.ExcelFile(file_like)
        col_name=table.parse(0).icol(1).real[1:]
        col_datetime=table.parse(0).icol(3)[1:]
        

        col_date=[]
        col_time=[]
        for c in col_datetime:
            col_date.append(c[:10])
            col_time.append(c[11:])
        date_unique=list(set(col_date))

        date_unique_copy=date_unique # delete the possible holiday which workers are less than 4 presons;
        for date_i in date_unique_copy:
            index_tmp=[i for i, key in enumerate(col_date) if key == date_i]
            date_name=[col_name[j] for j in index_tmp]
            if len(set(date_name))<=3:
                date_unique.remove(date_i)

        name_unique=list(set(col_name))
        duty_on=[]
        duty_off=[]
        for i in range(len(col_time)):
            if col_time[i]<'08:31:00':
                duty_on.append(i)
            elif col_time[i]>'17:00:00':
                duty_off.append(i)

        duty_on_record=[] # 创建存储每个人的上班打卡的所有日期;duty_on_record记录每个人上班打开正常的所有日期;duty_on_record_copy存储每个人(第一个元素是姓名)打没打卡的情况;
        duty_on_record_copy=[]
        for i in range(len(name_unique)):
            duty_on_record.append([])
            duty_on_record_copy.append([])
            duty_on_record_copy[i].append(name_unique[i])
        

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
                   duty_on_record_copy[i].append(date)
        tmp=[]
        for i in range(len(duty_on_record_copy)):
            if len(duty_on_record_copy[i])>1:
                tmp.append(' '.join(duty_on_record_copy[i]))
        self.duty_on='\n'.join(tmp) #'\n'.join(duty_on_output)


        duty_on_record=[] # 创建存储每个人的上班打卡的所有日期;duty_on_record记录每个人上班打开正常的所有日期;duty_on_record_copy存储每个人(第一个元素是姓名)打没打卡的情况;
        duty_on_record_copy=[]
        for i in range(len(name_unique)):
            duty_on_record.append([])
            duty_on_record_copy.append([])
            duty_on_record_copy[i].append(name_unique[i])
        

        for i in duty_off:
            for index in range(len(name_unique)):
                if col_name[i]==name_unique[index]:
                    duty_on_record[index].append(col_date[i])
                    break

        duty_on_output=[]
        for i in range(len(duty_on_record)):
           for date in date_unique:
               if date not in duty_on_record[i]:
                   #duty_on_output.append(','.join([name_unique[i],date]))
                   duty_on_record_copy[i].append(date)
        tmp=[]
        for i in range(len(duty_on_record_copy)):
            if len(duty_on_record_copy[i])>1:
                tmp.append(' '.join(duty_on_record_copy[i]))
        self.duty_off='\n'.join(tmp) #'\n'.join(duty_on_output)


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
    



    




