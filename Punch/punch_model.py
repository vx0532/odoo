# -*- coding: utf-8 -*-
from openerp import models, fields, exceptions
import base64

class PunchTask(models.Model):
    _name = 'punch.task'
    _description = 'Punch task'
    datafile=fields.Binary("File")
    records=fields.Text('有误记录')
    
    #@api.onchange("datafile")
    def action_upload(self,cr,uid,ids,context=None):
        this=self.browse(cr,uid,ids[0])
        data = base64.decodestring(this.datafile)
        this.records=data
        #raise exceptions.Warning(data)

        

    




