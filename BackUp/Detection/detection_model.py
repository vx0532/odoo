# -*- coding: utf-8 -*-
from openerp import models, fields,api
import base64

class DetectionTask(models.Model):
    _name = 'detection.task'
    _description = 'Detection task'
    _inherit = ['mail.thread']

    user_id=fields.Many2one('res.users','协调者')

    NO = fields.Char('编号')
    Lot = fields.Char('批号')
    Pro=fields.Char('蛋白含量（%）')
    H2O=fields.Char('水分（%）')
    CPC=fields.Char('CPC（%）')
    Metal=fields.Char('重金属（%）')
    Doc=fields.Integer("text")
    
    datafile=fields.Binary("File")
    
    #@api.onchange("datafile")
    def action_upload(self,cr,uid,ids,context=None):
        this=self.browse(cr,uid,ids[0])
        data = base64.decodestring(this.datafile)
        

    




