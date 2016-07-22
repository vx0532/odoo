# -*- coding: utf-8 -*-
from openerp import models, fields,api

class DetectionTask(models.Model):
    _name = 'detection.task'
    _description = 'Detection task'
    #_inherit = ['mail.thread']

    NO = fields.Char('编号')
    Lot = fields.Char('批号')
    Pro=fields.Char('蛋白含量（%）')
    pro=fields.Boolean('蛋白含量(%)')
    H2O=fields.Char('水分（%）')
    h2o=fields.Boolean('水分(%)')
    CPC=fields.Char('CPC（%）')
    cpc=fields.Boolean('CPC(%)')
    Metal=fields.Char('重金属（%）')
    metal=fields.Boolean('重金属(%)')
    Active=fields.Boolean('确认收样',default=False)
    Applicant=fields.Many2one('res.users','请验人:')#, compute='for_Applicant'
    Receiver=fields.Many2one('res.users', string='收样人:')#,compute='_for_Receiver'

    @api.onchange("NO","Lot")
    def for_Applicant(self):
        if self.NO:
            self.Applicant=self.env.uid

    @api.onchange("Active")
    def for_Receiver(self):
        if self.Active:
            self.Receiver=self.env.uid

'''
    @api.depends('Active')
    def _for_Receiver(self):
        self.Receiver=self.env.uid
        return {
            'warning':{
                'title':"确认收样",
                'message':"收样成功!",
            }
        }
'''   

    




