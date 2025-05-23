from odoo import api, fields, models


class PosReceiptLayout(models.TransientModel):
    _name = 'pos.receipt.layout'
    _description = 'POS Receipt Layout'

    def _get_default_pos_config(self):
        return self.env['pos.config'].browse([self.env.context.get('active_pos_config_id')])

    pos_config_id = fields.Many2one('pos.config', string='Point of Sale', default=_get_default_pos_config, required=True)
    receipt_layout = fields.Selection(related='pos_config_id.receipt_layout', readonly=False, required=True)
    receipt_logo = fields.Binary(related='pos_config_id.receipt_logo', readonly=False)
    receipt_header = fields.Text(related='pos_config_id.receipt_header', readonly=False)
    receipt_footer = fields.Text(related='pos_config_id.receipt_footer', readonly=False)
    is_restaurant = fields.Boolean(related='pos_config_id.module_pos_restaurant')
    receipt_preview = fields.Html(compute='_compute_receipt_preview')

    @api.depends('receipt_layout', 'receipt_header','receipt_footer','receipt_logo')
    def _compute_receipt_preview(self):
            if self.receipt_layout == "lined":
                self.receipt_preview = self.env['ir.ui.view']._render_template(
                    'pos_receipt_customizer.custom_receipt_lined',
                    {
                        'header': self.receipt_header,
                        'footer': self.receipt_footer,
                        'logo': self.receipt_logo,
                        'is_restaurant':self.is_restaurant
                    }
                )
            
            elif self.receipt_layout == "boxes":
                self.receipt_preview = self.env['ir.ui.view']._render_template(
                    'pos_receipt_customizer.custom_pos_receipt_boxed',
                    {
                        'header': self.receipt_header,
                        'footer': self.receipt_footer,
                        'logo': self.receipt_logo,
                        'is_restaurant':self.is_restaurant
                    }
                )

            else:
                self.receipt_preview = self.env['ir.ui.view']._render_template(
                    'pos_receipt_customizer.custom_receipt_static_light',
                    {
                        'header': self.receipt_header,
                        'footer': self.receipt_footer,
                        'logo': self.receipt_logo,
                        'is_restaurant':self.is_restaurant
                    }
                )

    def receipt_layout_save(self):
        return {'type': 'ir.actions.act_window_close'}
